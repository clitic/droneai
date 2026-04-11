import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Grouped classes: 14 original → 5 broader categories
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Normal",       # 0: NormalVideos
    "Violence",     # 1: Abuse, Arrest, Assault, Fighting
    "Theft",        # 2: Burglary, Robbery, Shoplifting, Stealing
    "Destruction",  # 3: Arson, Explosion, Shooting
    "Other",        # 4: RoadAccidents, Vandalism
]

_ORIGINAL_TO_GROUP = {
    "Normal":         0,
    "NormalVideos":   0,
    "Abuse":          1,
    "Arrest":         1,
    "Assault":        1,
    "Fighting":       1,
    "Burglary":       2,
    "Robbery":        2,
    "Shoplifting":    2,
    "Stealing":       2,
    "Arson":          3,
    "Explosion":      3,
    "Shooting":       3,
    "RoadAccidents":  4,
    "Vandalism":      4,
}


def scan_features(features_dir: Path) -> dict[str, list[tuple[str, int]]]:
    splits: dict[str, list[tuple[str, int]]] = {"train": [], "test": []}
    for split_name in ["Train", "Test"]:
        split_dir = features_dir / split_name
        if not split_dir.exists():
            continue
        for cat_dir in sorted(split_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            cls_idx = _ORIGINAL_TO_GROUP.get(cat_dir.name, -1)
            if cls_idx < 0:
                continue
            for npy in sorted(cat_dir.glob("*.npy")):
                splits[split_name.lower()].append((str(npy), cls_idx))
    return splits


class AnomalyDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], seq_len: int = 64) -> None:
        self.seq_len = seq_len
        self.samples = samples
        if not self.samples:
            raise ValueError("No samples found. Run embed.py first.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        feat = np.load(path)
        T, D = feat.shape
        if T >= self.seq_len:
            feat = feat[np.linspace(0, T - 1, self.seq_len, dtype=int)]
        else:
            feat = np.concatenate([feat, np.zeros((self.seq_len - T, D), dtype=feat.dtype)])
        return torch.from_numpy(feat).float(), torch.tensor(label, dtype=torch.long)


class AnomalyGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, bidirectional: bool = True,
                 num_classes: int = 5) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        d = hidden_size * (2 if bidirectional else 1)
        self.attn = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, 64),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        w = torch.softmax(self.attn(out), dim=1)
        return self.head(torch.sum(out * w, dim=1))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += criterion(logits, y).item() * x.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    return total / len(loader.dataset), acc, all_preds, all_labels


def main() -> None:
    epochs = 50
    features_dir = Path("datasets/ucf-crime-features")
    if not features_dir.exists():
        print(f"[ERROR] {features_dir} not found. Run embed.py first.")
        sys.exit(1)

    splits = scan_features(features_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_NAMES)

    train_ds = AnomalyDataset(splits["train"], seq_len=64)
    test_ds = AnomalyDataset(splits["test"], seq_len=64)

    train_counts = [0] * num_classes
    for _, lbl in train_ds.samples:
        train_counts[lbl] += 1

    total_samples = sum(train_counts)
    weights = torch.tensor(
        [total_samples / (num_classes * c) if c > 0 else 0.0 for c in train_counts],
        dtype=torch.float32, device=device
    )

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=32, num_workers=4, pin_memory=True)

    input_dim = train_ds[0][0].shape[-1]
    model = AnomalyGRU(input_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    save_dir = Path("runs")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    pbar = tqdm(range(1, epochs + 1), desc="Training", unit="ep", ncols=120)
    for ep in pbar:
        tl = train_epoch(model, train_dl, criterion, optimizer, device)
        vl, va, _, _ = evaluate(model, test_dl, criterion, device)
        scheduler.step()

        is_best = va > best_acc
        if is_best:
            best_acc = va
            torch.save({
                "epoch": ep, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "best_acc": best_acc,
                "input_dim": input_dim, "hidden_size": 128, "num_layers": 2,
                "dropout": 0.3, "bidirectional": True, "seq_len": 64,
                "num_classes": num_classes, "class_names": CLASS_NAMES,
            }, save_dir / "gru_best.pt")

        best_tag = " [BEST]" if is_best else ""
        tqdm.write(
            f"  Epoch {ep:3d}/{epochs} | TL={tl:.4f} VL={vl:.4f} "
            f"Acc={va:.4f} LR={optimizer.param_groups[0]['lr']:.1e}{best_tag}"
        )

    ckpt = torch.load(save_dir / "gru_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    _, fa, all_preds, all_labels = evaluate(model, test_dl, criterion, device)
    print(f"\n  Best Accuracy: {fa:.4f}")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))


if __name__ == "__main__":
    main()
