"""
Stage 3 — Train a GRU-based temporal classifier on extracted feature
embeddings for anomaly detection.

Reads .npy feature files produced by extract_features.py and trains a
binary classifier (Normal vs. Anomaly).

Usage:
    uv run python train_classifier.py
    uv run python train_classifier.py --epochs 30 --hidden-size 128 --num-layers 2
    uv run python train_classifier.py --help
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AnomalyDataset(Dataset):
    """PyTorch Dataset that loads .npy feature sequences.

    Each sample is a (features, label) pair where features has shape
    (seq_len, embedding_dim) and label is 0 (Normal) or 1 (Anomaly).
    """

    def __init__(
        self,
        manifest: dict,
        split: str,
        seq_len: int = 64,
    ) -> None:
        self.seq_len = seq_len
        self.samples: list[tuple[str, int]] = []

        for key, info in manifest.items():
            if info["split"] == split:
                self.samples.append((info["npy_path"], info["label"]))

        if not self.samples:
            raise ValueError(
                f"No samples found for split '{split}'. "
                "Check that extract_features.py has been run."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        npy_path, label = self.samples[idx]
        features = np.load(npy_path)  # (T, D)

        # Pad or truncate to fixed seq_len
        T, D = features.shape
        if T >= self.seq_len:
            # Uniformly subsample
            indices = np.linspace(0, T - 1, self.seq_len, dtype=int)
            features = features[indices]
        else:
            # Zero-pad
            pad = np.zeros((self.seq_len - T, D), dtype=features.dtype)
            features = np.concatenate([features, pad], axis=0)

        return (
            torch.from_numpy(features).float(),
            torch.tensor(label, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AnomalyGRU(nn.Module):
    """GRU-based binary classifier for temporal anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * direction_factor, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * direction_factor),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * direction_factor, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, 1)
        """
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden*directions)

        # Attention pooling over time
        attn_weights = self.attention(gru_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(gru_out * attn_weights, dim=1)  # (batch, hidden*dir)

        logits = self.classifier(context)  # (batch, 1)
        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: AnomalyGRU,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item() * features.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: AnomalyGRU,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Returns (loss, accuracy, auc_roc)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(features)
        loss = criterion(logits, labels)
        total_loss += loss.item() * features.size(0)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0  # Only one class present

    return avg_loss, accuracy, auc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train GRU anomaly classifier on extracted features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--features-dir",
        type=str,
        default="features",
        help="Directory with .npy feature files and manifest.json",
    )
    p.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--seq-len", type=int, default=64, help="Fixed sequence length")
    p.add_argument(
        "--hidden-size", type=int, default=128, help="GRU hidden size"
    )
    p.add_argument("--num-layers", type=int, default=2, help="Number of GRU layers")
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    p.add_argument(
        "--bidirectional", action="store_true", default=True, help="Use bidirectional GRU"
    )
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument(
        "--weight-decay", type=float, default=1e-5, help="AdamW weight decay"
    )
    p.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cuda, cpu)"
    )
    p.add_argument(
        "--save-dir", type=str, default="models", help="Directory to save model"
    )
    p.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load manifest
    manifest_path = Path(args.features_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run extract_features.py first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    print("=" * 60)
    print("  DroneAI — Stage 3: GRU Anomaly Classifier Training")
    print("=" * 60)
    print(f"  Features: {args.features_dir}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Seq len:  {args.seq_len}")
    print(f"  Hidden:   {args.hidden_size}")
    print(f"  Layers:   {args.num_layers}")
    print(f"  LR:       {args.lr}")
    print(f"  Device:   {device}")
    print("=" * 60)

    # Datasets
    train_ds = AnomalyDataset(manifest, split="train", seq_len=args.seq_len)
    test_ds = AnomalyDataset(manifest, split="test", seq_len=args.seq_len)

    # Count class distribution
    train_labels = [s[1] for s in train_ds.samples]
    n_normal = train_labels.count(0)
    n_anomaly = train_labels.count(1)
    print(f"\n  Train: {len(train_ds)} clips ({n_normal} normal, {n_anomaly} anomaly)")
    print(f"  Test:  {len(test_ds)} clips")

    # Weighted loss for class imbalance
    if n_normal > 0 and n_anomaly > 0:
        pos_weight = torch.tensor([n_normal / n_anomaly], device=device)
    else:
        pos_weight = torch.tensor([1.0], device=device)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=True,
    )

    # Auto-detect embedding dim from first sample
    sample_features, _ = train_ds[0]
    input_dim = sample_features.shape[-1]
    print(f"  Embedding dim: {input_dim}\n")

    # Model
    model = AnomalyGRU(
        input_dim=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,} (trainable: {trainable_params:,})\n")

    # Optimizer & scheduler
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Training loop
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:3d}/{args.epochs} │ "
            f"Train Loss: {train_loss:.4f} │ "
            f"Val Loss: {val_loss:.4f} │ "
            f"Val Acc: {val_acc:.4f} │ "
            f"Val AUC: {val_auc:.4f} │ "
            f"LR: {current_lr:.2e}"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "input_dim": input_dim,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "bidirectional": args.bidirectional,
                "seq_len": args.seq_len,
            }
            torch.save(checkpoint, save_dir / "gru_best.pt")
            print(f"           └─ ✅ New best model saved (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  ⏹  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Final evaluation with best model
    print("\n" + "=" * 60)
    print("  Final Evaluation on Test Set (Best Model)")
    print("=" * 60)

    checkpoint = torch.save  # Dummy reference to avoid lint
    best_ckpt = torch.load(save_dir / "gru_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])

    _, final_acc, final_auc = evaluate(model, test_loader, criterion, device)

    # Detailed classification report
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().flatten().tolist())

    print(f"\n  Accuracy: {final_acc:.4f}")
    print(f"  AUC-ROC:  {final_auc:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['Normal', 'Anomaly'])}")
    print(f"\n  Best model saved to: {save_dir / 'gru_best.pt'}")
    print("\n🚀 Pipeline complete! Use inference.py or app.py to run predictions.")


if __name__ == "__main__":
    main()
