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


def quantize_predictions(predictions, step):
    if step <= 0:
        return np.asarray(predictions, dtype=np.float64)
    return np.round(np.asarray(predictions, dtype=np.float64) / step) * step


def fit_channel_norm(X):
    # X = N x C x T. Per-channel normalization.
    mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = X.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def apply_channel_norm(X, mean, std, clip=5.0):
    X = (X - mean) / std
    return np.clip(X, -clip, clip).astype(np.float32)


def apply_epoch_norm(X):
    mean = X.mean(axis=2, keepdims=True)
    std = np.maximum(X.std(axis=2, keepdims=True), 1e-6)
    return ((X - mean) / std).astype(np.float32)


def fit_signal_norm(features):
    mean = features.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(features.std(axis=0, keepdims=True), 1e-6).astype(np.float32)
    return mean, std


def apply_signal_norm(features, mean, std):
    return ((features - mean) / std).astype(np.float32)


def make_signal_features(X, sfreq):
    """Compute compact per-epoch features directly from EEG."""
    X = np.asarray(X, dtype=np.float32)
    centered = X - X.mean(axis=2, keepdims=True)
    rms = np.sqrt(np.mean(centered ** 2, axis=(1, 2)))
    spread = centered.std(axis=(1, 2))
    abs_peak = np.max(np.abs(centered), axis=(1, 2))
    zero_crossings = np.mean(
        centered[:, :, 1:] * centered[:, :, :-1] < 0, axis=(1, 2)
    )
    spectrum = np.abs(np.fft.rfft(centered, axis=2)) ** 2
    frequencies = np.fft.rfftfreq(X.shape[2], 1.0 / sfreq)
    bands = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
    band_power = [
        spectrum[:, :, (frequencies >= low) & (frequencies < high)].mean(axis=(1, 2))
        for low, high in bands
    ]
    return np.column_stack([rms, spread, abs_peak, zero_crossings, *band_power]).astype(np.float32)


class EEGNet(nn.Module):
    """Compact EEGNet-like model for N x 1 x C x T input."""
    def __init__(self, n_channels, n_times, n_outputs=1,
                 temporal_filters=12, depth_multiplier=3, dropout=0.35,
                 extra_temporal_block=False, multi_scale_temporal=False,
                 signal_feature_dim=0, ordinal_bins=0):
        super().__init__()
        F1 = temporal_filters
        D = depth_multiplier
        temporal_channels = F1 * 3 if multi_scale_temporal else F1
        F2 = temporal_channels * D

        # Same-padding temporal convolution keeps time dimension unchanged.
        if multi_scale_temporal:
            self.temporal = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(1, F1, kernel_size=(1, kernel),
                              padding=(0, kernel // 2), bias=False),
                    nn.BatchNorm2d(F1),
                )
                for kernel in (8, 16, 32)
            ])
        else:
            self.temporal = nn.Sequential(
                nn.Conv2d(1, F1, kernel_size=(1, 32), padding=(0, 16), bias=False),
                nn.BatchNorm2d(F1),
            )

        # Depthwise spatial convolution: each temporal filter gets its own
        # channel-spanning spatial filter.
        self.spatial = nn.Sequential(
            nn.Conv2d(
                temporal_channels, temporal_channels * D,
                kernel_size=(n_channels, 1), groups=temporal_channels,
                bias=False
            ),
            nn.BatchNorm2d(temporal_channels * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # Separable temporal convolution.
        separable_layers = [
            nn.Conv2d(
                F2, F2, kernel_size=(1, 16),
                padding=(0, 8), groups=F2, bias=False
            ),
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
        ]
        if extra_temporal_block:
            separable_layers.extend([
                nn.Conv2d(
                    F2, F2, kernel_size=(1, 8), padding=(0, 4),
                    groups=F2, bias=False
                ),
                nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
            ])
        separable_layers.extend([
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        ])
        self.separable = nn.Sequential(*separable_layers)

        # Infer flattened dimension instead of hard-coding it.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat_dim = self._features(dummy).flatten(1).shape[1]

        self.signal_features = None
        classifier_input = flat_dim
        if signal_feature_dim:
            self.signal_features = nn.Sequential(
                nn.Linear(signal_feature_dim, 16),
                nn.LayerNorm(16),
                nn.ELU(),
                nn.Dropout(dropout),
            )
            classifier_input += 16
        self.classifier = nn.Linear(classifier_input, n_outputs)
        self.ordinal_head = (
            nn.Linear(classifier_input, ordinal_bins) if ordinal_bins else None
        )

    def _features(self, x):
        if isinstance(self.temporal, nn.ModuleList):
            x = torch.cat([branch(x) for branch in self.temporal], dim=1)
        else:
            x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return x

    def forward(self, x, signal_features=None):
        features = [self._features(x).flatten(1)]
        if self.signal_features is not None:
            features.append(self.signal_features(signal_features))
        combined = torch.cat(features, dim=1)
        prediction = self.classifier(combined).squeeze(1)
        if self.ordinal_head is not None:
            return prediction, self.ordinal_head(combined)
        return prediction


def make_loader(X, y, batch_size, shuffle, signal_features=None):
    tensors = [torch.from_numpy(X[:, None, :, :])]
    if signal_features is not None:
        tensors.append(torch.from_numpy(signal_features))
    tensors.append(torch.from_numpy(y))
    ds = TensorDataset(
        *tensors,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def train_one_fold(Xtr, ytr, Xva, yva, signal_tr, signal_va, args, device):
    if args.epoch_normalize:
        Xtr = apply_epoch_norm(Xtr)
        Xva = apply_epoch_norm(Xva)
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
        extra_temporal_block=args.extra_temporal_block,
        multi_scale_temporal=args.multi_scale_temporal,
        signal_feature_dim=signal_tr.shape[1] if signal_tr is not None else 0,
        ordinal_bins=14 if args.ordinal_auxiliary else 0,
    ).to(device)

    if args.loss == "huber":
        criterion = nn.HuberLoss(delta=1.0)
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    elif args.loss == "epsilon":
        criterion = None
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")
    ordinal_criterion = nn.BCEWithLogitsLoss()
    ordinal_thresholds = torch.arange(
        1.125, 4.5, 0.25, device=device, dtype=torch.float32
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    tr_loader = make_loader(Xtr, ytr, args.batch_size, True, signal_tr)
    va_loader = make_loader(Xva, yva, args.batch_size, False, signal_va)

    best_state, best_mae, best_epoch = None, np.inf, -1
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tr_loader:
            xb, signal_batch, yb = batch if signal_tr is not None else (*batch[:1], None, batch[1])
            xb = xb.to(device)
            if signal_batch is not None:
                signal_batch = signal_batch.to(device).float()
            yb = yb.to(device).float().view(-1, 1)
            if args.input_noise > 0:
                xb = xb + torch.randn_like(xb) * args.input_noise
            if args.channel_dropout > 0:
                keep = (torch.rand(
                    xb.shape[0], 1, xb.shape[2], 1, device=xb.device
                ) >= args.channel_dropout).to(xb.dtype)
                xb = xb * keep / (1.0 - args.channel_dropout)
            if args.time_jitter > 0:
                shifts = torch.randint(
                    -args.time_jitter, args.time_jitter + 1,
                    (xb.shape[0],), device=xb.device
                )
                xb = torch.stack([
                    torch.roll(sample, int(shift.item()), dims=-1)
                    for sample, shift in zip(xb, shifts)
                ])
            optimizer.zero_grad(set_to_none=True)
            output = model(xb, signal_batch)
            if args.ordinal_auxiliary:
                pred, ordinal_logits = output
                ordinal_target = (yb > ordinal_thresholds).float()
                loss = criterion(pred, yb.squeeze(1)) + (
                    args.ordinal_loss_weight
                    * ordinal_criterion(ordinal_logits, ordinal_target)
                )
            else:
                pred = output
                if args.loss == "epsilon":
                    error = torch.abs(pred - yb.squeeze(1)) - args.margin
                    loss = torch.relu(error).pow(2).mean()
                else:
                    loss = criterion(pred, yb.squeeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        yp, yt = [], []
        with torch.no_grad():
            for batch in va_loader:
                xb, signal_batch, yb = batch if signal_va is not None else (*batch[:1], None, batch[1])
                signal_batch = signal_batch.to(device).float() if signal_batch is not None else None
                output = model(xb.to(device), signal_batch)
                pred = output[0] if args.ordinal_auxiliary else output
                pred = pred.cpu().numpy()
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
def predict(model, X, mean, std, clip, batch_size, device, signal_features=None):
    X = apply_channel_norm(X, mean, std, clip)
    loader = make_loader(
        X, np.zeros(len(X), dtype=np.float32), batch_size, False, signal_features
    )
    model.eval()
    out = []
    for batch in loader:
        xb, signal_batch, _ = batch if signal_features is not None else (*batch[:1], None, batch[1])
        signal_batch = signal_batch.to(device).float() if signal_batch is not None else None
        output = model(xb.to(device), signal_batch)
        prediction = output[0] if model.ordinal_head is not None else output
        out.append(prediction.cpu().numpy())
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
    ap.add_argument("--extra-temporal-block", action="store_true")
    ap.add_argument("--multi-scale-temporal", action="store_true")
    ap.add_argument("--signal-feature-branch", action="store_true")
    ap.add_argument("--ordinal-auxiliary", action="store_true")
    ap.add_argument("--ordinal-loss-weight", type=float, default=0.15)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--epoch-normalize", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--input-noise", type=float, default=0.0)
    ap.add_argument("--channel-dropout", type=float, default=0.0)
    ap.add_argument("--time-jitter", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--quantize-step", type=float, default=0.5)
    ap.add_argument("--target-step", type=float, default=0.0)
    ap.add_argument("--loss", choices=["huber", "mae", "epsilon"], default="huber")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    seed_everything(a.seed)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    z = np.load(a.input, allow_pickle=True)
    X = z["X"].astype(np.float32)
    power = z["power"].astype(float)
    groups = z["global_subject"].astype(str)
    signal_features = (
        make_signal_features(X, float(z["sfreq"]))
        if a.signal_feature_branch else None
    )

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
    oof_raw_pred = np.full(len(X), -1.0, dtype=np.float64)
    oof_pred = np.full(len(X), -1.0, dtype=np.float64)
    oof_y = np.full(len(X), -1.0, dtype=np.float64)
    fold_id = np.full(len(X), -1, dtype=np.int64)

    for fold, (tr, va) in enumerate(
        splitter.split(X, global_y_for_split, groups=groups), start=1
    ):
        ytr = power[tr].astype(np.float32)
        yva = power[va].astype(np.float32)
        if signal_features is not None:
            signal_mean, signal_std = fit_signal_norm(signal_features[tr])
            signal_tr = apply_signal_norm(signal_features[tr], signal_mean, signal_std)
            signal_va = apply_signal_norm(signal_features[va], signal_mean, signal_std)
        else:
            signal_tr, signal_va = None, None

        train_target = quantize_predictions(ytr, a.target_step).astype(np.float32)
        model, mean, std, history, best_epoch = train_one_fold(
            X[tr], train_target, X[va], yva, signal_tr, signal_va, a, device
        )
        raw_pred = predict(
            model, X[va], mean, std, a.clip, a.batch_size, device, signal_va
        )
        pred = quantize_predictions(raw_pred, a.quantize_step)

        metrics = evaluate(yva, pred, groups[va], a.bootstrap, margin=a.margin)
        metrics.update({
            "fold": fold,
            "margin": float(a.margin),
            "quantize_step": float(a.quantize_step),
            "target_step": float(a.target_step),
            "best_epoch": best_epoch,
            "n_train_epochs": int(len(tr)),
            "n_val_epochs": int(len(va)),
            "n_train_subjects": int(len(np.unique(groups[tr]))),
            "n_val_subjects": int(len(np.unique(groups[va]))),
        })
        all_fold_metrics.append(metrics)

        oof_raw_pred[va] = raw_pred
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
        "quantize_step": float(a.quantize_step),
        "target_step": float(a.target_step),
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
        "raw_prediction": oof_raw_pred,
        "prediction": oof_pred,
        "fold": fold_id,
    }).to_csv(outdir / "oof_predictions.csv", index=False)

    print("\nOOF:")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
