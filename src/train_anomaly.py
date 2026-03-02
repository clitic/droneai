"""
Stage 3 -- Train a bidirectional GRU for anomaly detection
on extracted embeddings (Normal vs Anomaly).

Usage:
    uv run python src/train_anomaly.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AnomalyDataset(Dataset):
    def __init__(self, manifest: dict, split: str, seq_len: int = 64) -> None:
        self.seq_len = seq_len
        self.samples = [(v["npy_path"], v["label"]) for v in manifest.values() if v["split"] == split]
        if not self.samples:
            raise ValueError(f"No samples for '{split}'. Run extract_embeddings.py first.")

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
        return torch.from_numpy(feat).float(), torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AnomalyGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, bidirectional: bool = True) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        d = hidden_size * (2 if bidirectional else 1)
        self.attn = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, 64),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        w = torch.softmax(self.attn(out), dim=1)
        return self.head(torch.sum(out * w, dim=1))


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
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
    total, probs, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        logits = model(x)
        total += criterion(logits, y).item() * x.size(0)
        probs.extend(torch.sigmoid(logits).cpu().numpy().flatten().tolist())
        labels.extend(y.cpu().numpy().flatten().tolist())
    preds = [int(p >= 0.5) for p in probs]
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
    return total / len(loader.dataset), acc, auc


def main() -> None:
    mf = Path("features/manifest.json")
    if not mf.exists():
        print(f"[ERROR] {mf} not found. Run extract_embeddings.py first.")
        sys.exit(1)

    with open(mf) as f:
        manifest = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  DroneAI -- Stage 3: GRU Anomaly Classifier")
    print("=" * 60)
    print(f"  Device: {device}")
    print("=" * 60)

    train_ds = AnomalyDataset(manifest, "train", seq_len=64)
    test_ds = AnomalyDataset(manifest, "test", seq_len=64)

    n0 = sum(1 for _, l in train_ds.samples if l == 0)
    n1 = sum(1 for _, l in train_ds.samples if l == 1)
    print(f"\n  Train: {len(train_ds)} ({n0} normal, {n1} anomaly)")
    print(f"  Test:  {len(test_ds)}")

    pw = torch.tensor([n0 / n1] if n0 and n1 else [1.0], device=device)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=32, num_workers=4, pin_memory=True)

    input_dim = train_ds[0][0].shape[-1]
    print(f"  Embedding dim: {input_dim}\n")

    model = AnomalyGRU(input_dim).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    save_dir = Path("models")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_auc, patience = 0.0, 0

    for ep in range(1, 31):
        tl = train_epoch(model, train_dl, criterion, optimizer, device)
        vl, va, vauc = evaluate(model, test_dl, criterion, device)
        scheduler.step()

        print(f"  Epoch {ep:3d}/30 | Train: {tl:.4f} | Val: {vl:.4f} | "
              f"Acc: {va:.4f} | AUC: {vauc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if vauc > best_auc:
            best_auc, patience = vauc, 0
            torch.save({
                "epoch": ep, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "best_auc": best_auc,
                "input_dim": input_dim, "hidden_size": 128, "num_layers": 2,
                "dropout": 0.3, "bidirectional": True, "seq_len": 64,
            }, save_dir / "gru_best.pt")
            print(f"           +-- [BEST] AUC: {best_auc:.4f}")
        else:
            patience += 1
            if patience >= 10:
                print(f"\n  [STOP] Early stopping at epoch {ep}")
                break

    # Final eval
    print(f"\n{'='*60}\n  Final Evaluation\n{'='*60}")
    ckpt = torch.load(save_dir / "gru_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    _, fa, fauc = evaluate(model, test_dl, criterion, device)

    model.eval()
    ap, al = [], []
    with torch.no_grad():
        for x, y in test_dl:
            p = (torch.sigmoid(model(x.to(device))).cpu().numpy().flatten() >= 0.5).astype(int)
            ap.extend(p.tolist())
            al.extend(y.numpy().flatten().tolist())

    print(f"\n  Accuracy: {fa:.4f}\n  AUC-ROC:  {fauc:.4f}")
    print(f"\n{classification_report(al, ap, target_names=['Normal', 'Anomaly'])}")
    print(f"  Model: {save_dir / 'gru_best.pt'}")
    print("\n>> Done! Launch UI: uv run python src/app.py")


if __name__ == "__main__":
    main()
