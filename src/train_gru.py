import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CLASS_NAMES = [
    "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]

_CAT_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
_CAT_TO_IDX["NormalVideos"] = 0


def scan_features(features_dir: Path) -> dict[str, list[tuple[str, int]]]:
    splits: dict[str, list[tuple[str, int]]] = {"train": [], "test": []}
    for split_name in ["Train", "Test"]:
        split_dir = features_dir / split_name
        if not split_dir.exists():
            continue
        for cat_dir in sorted(split_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            cls_idx = _CAT_TO_IDX.get(cat_dir.name, -1)
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
    def __init__(self, input_dim: int, hidden_size: int = 256, num_layers: int = 3,
                 dropout: float = 0.4, bidirectional: bool = True,
                 num_classes: int = 14) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        d = hidden_size * (2 if bidirectional else 1)
        self.attn = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

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
    features_dir = Path("datasets/ucf-crime-features")
    if not features_dir.exists():
        print(f"[ERROR] {features_dir} not found. Run embed.py first.")
        sys.exit(1)

    splits = scan_features(features_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_NAMES)
    max_epochs = 100



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

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=64, num_workers=4, pin_memory=True)

    input_dim = train_ds[0][0].shape[-1]
    model = AnomalyGRU(input_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    save_dir = Path("runs")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_acc, patience = 0.0, 0

    epoch_bar = tqdm(range(1, max_epochs + 1), desc="Training", unit="ep")
    for ep in epoch_bar:
        tl = train_epoch(model, train_dl, criterion, optimizer, device)
        vl, va, _, _ = evaluate(model, test_dl, criterion, device)
        scheduler.step()

        marker = " (best)" if va > best_acc else ""
        epoch_bar.write(f"  ep {ep:3d} | loss {tl:.4f} | val {vl:.4f} | acc {va:.4f}{marker}")

        if va > best_acc:
            best_acc, patience = va, 0
            torch.save({
                "epoch": ep, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "best_acc": best_acc,
                "input_dim": input_dim, "hidden_size": 256, "num_layers": 3,
                "dropout": 0.4, "bidirectional": True, "seq_len": 64,
                "num_classes": num_classes, "class_names": CLASS_NAMES,
            }, save_dir / "gru_best.pt")
        else:
            patience += 1
            if patience >= 20:
                epoch_bar.close()
                break

    ckpt = torch.load(save_dir / "gru_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    _, fa, all_preds, all_labels = evaluate(model, test_dl, criterion, device)
    print(f"\n{classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0)}")


if __name__ == "__main__":
    main()
