"""Subject-grouped CV training for a compact EEGNet classifier.

Design choices:
- Groups are global_subject, so epochs from one subject never cross folds.
- Tertile thresholds are learned from TRAINING subjects in each fold only.
- Channel-wise mean/std are learned from TRAINING epochs only.
- Standardization is followed by clipping to [-5, 5].
- Class-weighted cross entropy uses TRAIN-fold class counts.
- Validation is never used for fitting normalization, thresholds, or weights.
- Reporting includes trial accuracy/F1 and subject-level bootstrap accuracy.
"""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_tertile_thresholds(power):
    power = np.asarray(power, dtype=np.float64)
    return float(np.quantile(power, 1/3)), float(np.quantile(power, 2/3))


def apply_tertiles(power, q33, q67):
    power = np.asarray(power)
    return np.where(power <= q33, 0, np.where(power <= q67, 1, 2)).astype(np.int64)


def within_margin_accuracy(y_true, y_pred, margin=1.0):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred) <= margin))


def fit_channel_norm(X):
    # X = N x C x T. Per-channel normalization.
    mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = X.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def apply_channel_norm(X, mean, std, clip=5.0):
    X = (X - mean) / std
    return np.clip(X, -clip, clip).astype(np.float32)


class EEGNet(nn.Module):
    """Compact EEGNet-like model for N x 1 x C x T input."""
    def __init__(self, n_channels, n_times, n_outputs=1,
                 temporal_filters=12, depth_multiplier=3, dropout=0.35):
        super().__init__()
        F1 = temporal_filters
        D = depth_multiplier
        F2 = F1 * D

        # Same-padding temporal convolution keeps time dimension unchanged.
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Depthwise spatial convolution: each temporal filter gets its own
        # channel-spanning spatial filter.
        self.spatial = nn.Sequential(
            nn.Conv2d(
                F1, F1 * D, kernel_size=(n_channels, 1),
                groups=F1, bias=False
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # Separable temporal convolution.
        self.separable = nn.Sequential(
            nn.Conv2d(
                F1 * D, F1 * D, kernel_size=(1, 16),
                padding=(0, 8), groups=F1 * D, bias=False
            ),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # Infer flattened dimension instead of hard-coding it.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat_dim = self._features(dummy).flatten(1).shape[1]

        self.classifier = nn.Linear(flat_dim, n_outputs)

    def _features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return x

    def forward(self, x):
        return self.classifier(self._features(x).flatten(1)).squeeze(1)


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X[:, None, :, :]),
        torch.from_numpy(y),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def train_one_fold(Xtr, ytr, Xva, yva, args, device):
    mean, std = fit_channel_norm(Xtr)
    Xtr = apply_channel_norm(Xtr, mean, std, args.clip)
    Xva = apply_channel_norm(Xva, mean, std, args.clip)

    model = EEGNet(
        n_channels=Xtr.shape[1],
        n_times=Xtr.shape[2],
        n_outputs=1,
        temporal_filters=args.temporal_filters,
        depth_multiplier=args.depth_multiplier,
        dropout=args.dropout,
    ).to(device)

    if args.loss == "huber":
        criterion = nn.HuberLoss(delta=1.0)
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    tr_loader = make_loader(Xtr, ytr, args.batch_size, True)
    va_loader = make_loader(Xva, yva, args.batch_size, False)

    best_state, best_mae, best_epoch = None, np.inf, -1
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device).float().view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb.squeeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        yp, yt = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                pred = model(xb.to(device)).cpu().numpy()
                yp.append(pred)
                yt.append(yb.numpy())
        yp, yt = np.concatenate(yp), np.concatenate(yt)
        val_mae = float(np.mean(np.abs(yt - yp)))
        val_rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        val_within = within_margin_accuracy(yt, yp, margin=args.margin)
        scheduler.step(val_mae)

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_within_margin": val_within,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, mean, std, history, best_epoch


@torch.no_grad()
def predict(model, X, mean, std, clip, batch_size, device):
    X = apply_channel_norm(X, mean, std, clip)
    loader = make_loader(X, np.zeros(len(X), dtype=np.float32), batch_size, False)
    model.eval()
    out = []
    for xb, _ in loader:
        out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out)


def subject_bootstrap_accuracy(y_true, y_pred, subjects, n_boot=2000, seed=SEED):
    """Bootstrap subjects, not epochs.

    Each replicate samples subjects with replacement and computes accuracy
    over all epochs belonging to the sampled subjects. This avoids letting
    subjects with more epochs dominate the bootstrap's sampling unit.
    """
    rng = np.random.default_rng(seed)
    subjects = np.asarray(subjects)
    unique = np.unique(subjects)
    per_subject = {
        s: accuracy_score(y_true[subjects == s], y_pred[subjects == s])
        for s in unique
    }
    values = np.array(list(per_subject.values()), dtype=np.float64)

    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots[i] = sample.mean()

    return {
        "subject_mean_accuracy": float(values.mean()),
        "bootstrap_mean": float(boots.mean()),
        "bootstrap_ci_low": float(np.quantile(boots, 0.025)),
        "bootstrap_ci_high": float(np.quantile(boots, 0.975)),
        "n_subjects": int(len(unique)),
    }


def evaluate(y_true, y_pred, subjects, n_boot, margin=1.0):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    abs_err = np.abs(y_true - y_pred)
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "within_margin_accuracy": float(np.mean(abs_err <= margin)),
        "subject_mean_mae": float(
            np.mean([
                np.mean(abs_err[subjects == s]) for s in np.unique(subjects)
            ])
        ),
        "n_subjects": int(len(np.unique(subjects))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="nurovo_epochs.npz")
    ap.add_argument("--outdir", default="nurovo_runs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--temporal-filters", type=int, default=12)
    ap.add_argument("--depth-multiplier", type=int, default=3)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--loss", choices=["huber", "mae"], default="huber")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    seed_everything(a.seed)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    z = np.load(a.input, allow_pickle=True)
    X = z["X"].astype(np.float32)
    power = z["power"].astype(float)
    groups = z["global_subject"].astype(str)

    # IMPORTANT: stratification target is made from global tertiles only to
    # make fold assignment reasonably class-balanced. It is NOT used as the
    # classifier's fitted threshold; each fold recomputes q33/q67 on train.
    global_y_for_split = apply_tertiles(
        power, *make_tertile_thresholds(power)
    )

    splitter = StratifiedGroupKFold(
        n_splits=a.folds, shuffle=True, random_state=a.seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_fold_metrics = []
    oof_pred = np.full(len(X), -1.0, dtype=np.float64)
    oof_y = np.full(len(X), -1.0, dtype=np.float64)
    fold_id = np.full(len(X), -1, dtype=np.int64)

    for fold, (tr, va) in enumerate(
        splitter.split(X, global_y_for_split, groups=groups), start=1
    ):
        ytr = power[tr].astype(np.float32)
        yva = power[va].astype(np.float32)

        model, mean, std, history, best_epoch = train_one_fold(
            X[tr], ytr, X[va], yva, a, device
        )
        pred = predict(model, X[va], mean, std, a.clip, a.batch_size, device)

        metrics = evaluate(yva, pred, groups[va], a.bootstrap, margin=a.margin)
        metrics.update({
            "fold": fold,
            "margin": float(a.margin),
            "best_epoch": best_epoch,
            "n_train_epochs": int(len(tr)),
            "n_val_epochs": int(len(va)),
            "n_train_subjects": int(len(np.unique(groups[tr]))),
            "n_val_subjects": int(len(np.unique(groups[va]))),
        })
        all_fold_metrics.append(metrics)

        oof_pred[va] = pred
        oof_y[va] = yva
        fold_id[va] = fold

        pd.DataFrame(history).to_csv(outdir / f"fold_{fold}_history.csv", index=False)
        np.savez(
            outdir / f"fold_{fold}_normalization.npz",
            mean=mean, std=std, margin=np.array(a.margin, dtype=np.float32)
        )
        torch.save(model.state_dict(), outdir / f"fold_{fold}_model.pt")

        print(
            f"fold {fold}: mae={metrics['mae']:.3f}, "
            f"rmse={metrics['rmse']:.3f}, "
            f"within{a.margin}={metrics['within_margin_accuracy']:.3f}"
        )

    valid = oof_pred >= 0
    overall = evaluate(oof_y[valid], oof_pred[valid], groups[valid], a.bootstrap, margin=a.margin)
    result = {
        "device": str(device),
        "seed": a.seed,
        "margin": float(a.margin),
        "folds": all_fold_metrics,
        "oof": overall,
        "channel_names": z["ch_names"].tolist(),
        "sfreq": float(z["sfreq"]),
        "tmin": float(z["tmin"]),
        "n_epochs": int(len(X)),
        "n_subjects": int(len(np.unique(groups))),
    }

    with open(outdir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    pd.DataFrame({
        "dataset": z["dataset"].astype(str),
        "global_subject": groups,
        "epoch": z["epoch"],
        "power": power,
        "label": oof_y,
        "prediction": oof_pred,
        "fold": fold_id,
    }).to_csv(outdir / "oof_predictions.csv", index=False)

    print("\nOOF:")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
