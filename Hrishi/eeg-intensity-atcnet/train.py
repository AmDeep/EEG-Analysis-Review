"""
Train ATCNetRegressor (raw EEG) on the pooled processed datasets in data/processed/.

Split strategy: subject-wise (grouped) so no subject appears in both train and test --
use --loso for leave-one-subject-out (slow, most rigorous) or --split for a single
grouped train/test split (fast, for iterating).

Usage:
    python train.py --mode raw --data data/processed --epochs 100 --loso
    python train.py --mode raw --data data/processed --epochs 100 --split 0.2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from models.atcnet import ATCNetRegressor


def load_pooled(data_dir: Path):
    """Concatenate X/y/subjects across every dataset folder under data_dir, offsetting
    subject ids so subjects from different datasets never collide in a GroupKFold."""
    all_X, all_y, all_groups = [], [], []
    offset = 0
    for ds_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        x_path = ds_dir / "X.npy"
        if not x_path.exists():
            continue
        X = np.load(x_path)
        y = np.load(ds_dir / "y_relative.npy")  # use the harmonized [0,1] label for pooled training
        subj = np.load(ds_dir / "subjects.npy") + offset
        all_X.append(X)
        all_y.append(y)
        all_groups.append(subj)
        offset = subj.max() + 1
        print(f"  loaded {ds_dir.name}: {X.shape[0]} trials, {X.shape[1]} ch, {X.shape[2]} samples")

    if not all_X:
        raise FileNotFoundError(f"No processed datasets found under {data_dir}. Run preprocessing/build_epochs.py first.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    return X, y, groups


def train_one_fold(model, train_loader, val_loader, epochs, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(loss_fn(model(xb), yb).item())
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch:3d}  val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw"], default="raw")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loso", action="store_true", help="leave-one-subject-out CV (slow)")
    ap.add_argument("--split", type=float, default=0.2, help="held-out fraction if not --loso")
    ap.add_argument("--out", default="runs/atcnet_raw")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    X, y, groups = load_pooled(Path(args.data))
    n_channels, n_samples = X.shape[1], X.shape[2]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []

    if args.loso:
        splitter = GroupKFold(n_splits=len(np.unique(groups)))
        splits = list(splitter.split(X, y, groups))
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.split, random_state=0)
        splits = list(splitter.split(X, y, groups))

    best_overall = float("inf")
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        model = ATCNetRegressor(n_channels=n_channels, n_samples=n_samples).to(device)

        train_ds = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx]))
        val_ds = TensorDataset(torch.tensor(X[val_idx]), torch.tensor(y[val_idx]))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        print(f"fold {fold_i}: train={len(train_idx)} val={len(val_idx)}")
        model, val_loss = train_one_fold(model, train_loader, val_loader, args.epochs, args.lr, device)
        fold_metrics.append(val_loss)

        if val_loss < best_overall:
            best_overall = val_loss
            torch.save(model.state_dict(), out_dir / "best.pt")

        if not args.loso:
            break  # single split, no need to loop

    summary = {"fold_val_losses": fold_metrics, "mean_val_loss": float(np.mean(fold_metrics))}
    with open(out_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Training summary:", summary)


if __name__ == "__main__":
    main()
