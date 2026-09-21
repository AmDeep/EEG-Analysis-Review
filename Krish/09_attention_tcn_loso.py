#!/usr/bin/env python3
"""Attention-TCN for interval-level laser-intensity prediction with true LOSO CV.

Each LOSO fold holds out one dataset-specific participant.  The network sees a
causal window ending at the interval being predicted, so it never uses future
intervals.  It emits both a discrete set-point and an expected intensity in
Joules.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "Aditya" / "Feature Extraction" / "features_combined.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "attention_tcn_loso_results"
FEATURES = [
    "n2_amp", "n2_lat", "p2_amp", "p2_lat", "n2p2_amp", "gamma_power",
    "alpha_erd_pct", "beta_erd_pct", "psd_delta", "psd_theta", "psd_alpha",
    "psd_beta", "psd_gamma", "plv_Fz-Cz", "plv_Cz-Pz", "plv_C3-C4",
    "plv_FCz-CPz", "perm_entropy", "spectral_entropy", "sample_entropy",
    "higuchi_fd", "dfa", "hjorth_mobility", "hjorth_complexity",
]


@dataclass(frozen=True)
class Config:
    window: int = 16
    channels: int = 64
    levels: int = 4
    kernel_size: int = 3
    dropout: float = 0.20
    batch_size: int = 256
    epochs: int = 30
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.10
    seed: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Optional dataset IDs to include, for example: ds005293 ds005280")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fold-start", type=int, default=1,
                        help="First one-based LOSO fold to run (inclusive).")
    parser.add_argument("--fold-end", type=int, default=None,
                        help="Last one-based LOSO fold to run (inclusive).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run folds whose prediction fragment already exists.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(
    csv_path: Path,
    selected_datasets: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[float]]:
    data = pd.read_csv(csv_path)
    required = set(FEATURES + ["dataset", "subject", "epoch", "laser_power"])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if selected_datasets:
        available = set(data["dataset"].dropna().astype(str))
        unknown = sorted(set(selected_datasets).difference(available))
        if unknown:
            raise ValueError(f"Unknown dataset IDs: {unknown}. Available IDs: {sorted(available)}")
        data = data[data["dataset"].astype(str).isin(selected_datasets)].copy()

    data = data.dropna(subset=["dataset", "subject", "epoch", "laser_power"]).copy()
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    data["subject_id"] = data["dataset"].astype(str) + "__" + data["subject"].astype(str)
    # Stable temporal order is essential for constructing within-subject windows.
    data = data.sort_values(["subject_id", "epoch"], kind="mergesort").reset_index(drop=False)
    data = data.rename(columns={"index": "source_row"})

    intensity_values = sorted(data["laser_power"].astype(float).unique().tolist())
    class_lookup = {value: index for index, value in enumerate(intensity_values)}
    labels = data["laser_power"].astype(float).map(class_lookup).to_numpy(dtype=np.int64)
    groups = data["subject_id"].to_numpy()
    return data, labels, groups, intensity_values


def build_contexts(groups: np.ndarray, window: int) -> np.ndarray:
    """Return row indices for causal windows; -1 denotes left padding."""
    if window < 1:
        raise ValueError("--window must be at least 1")
    contexts = np.full((len(groups), window), -1, dtype=np.int64)
    start = 0
    while start < len(groups):
        stop = start + 1
        while stop < len(groups) and groups[stop] == groups[start]:
            stop += 1
        for row in range(start, stop):
            history_start = max(start, row - window + 1)
            history = np.arange(history_start, row + 1)
            contexts[row, -len(history):] = history
        start = stop
    return contexts


def fit_preprocessor(raw_features: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Median-impute and scale using training-participant statistics only."""
    train = raw_features[train_indices]
    medians = np.nanmedian(train, axis=0)
    # Defensive fallback for a feature absent throughout a training fold.
    medians = np.where(np.isnan(medians), 0.0, medians)
    imputed = np.where(np.isnan(raw_features), medians, raw_features)
    scaler = StandardScaler().fit(imputed[train_indices])
    transformed = scaler.transform(imputed).astype(np.float32)
    return transformed, medians, scaler


class WindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        contexts: np.ndarray,
        row_indices: np.ndarray,
    ) -> None:
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        selected_contexts = contexts[self.row_indices]
        self.valid = selected_contexts >= 0
        safe_contexts = np.where(self.valid, selected_contexts, 0)
        # Vectorized construction avoids rebuilding every window in Python on
        # every epoch, which is especially important for 678 LOSO fits.
        self.windows = features[safe_contexts].copy()
        self.windows[~self.valid] = 0.0
        self.labels = labels[self.row_indices]

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return (
            torch.from_numpy(self.windows[item]),
            torch.from_numpy(self.valid[item]),
            torch.tensor(self.labels[item], dtype=torch.long),
            int(self.row_indices[item]),
        )


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.trim = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.trim, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)
        return output[:, :, :-self.trim] if self.trim else output


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.dropout(self.activation(self.norm1(self.conv1(x))))
        x = self.dropout(self.activation(self.norm2(self.conv2(x))))
        return self.activation(x + residual)


class AttentionTCN(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        channels: int,
        levels: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        for level in range(levels):
            blocks.append(TemporalBlock(
                n_features if level == 0 else channels,
                channels,
                kernel_size,
                dilation=2 ** level,
                dropout=dropout,
            ))
        self.tcn = nn.Sequential(*blocks)
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.Tanh(),
            nn.Linear(channels // 2, 1),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, n_classes))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: batch x time x features; Conv1d expects batch x features x time.
        encoded = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        scores = self.attention(encoded).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return self.head(pooled), weights


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    contexts: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        WindowDataset(features, labels, contexts, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    intensity_tensor: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_rows, all_classes, all_expected, all_attention = [], [], [], []
    for features, mask, _, rows in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, attention = model(features.to(device), mask.to(device))
        probabilities = torch.softmax(logits.float(), dim=1)
        all_rows.append(rows.numpy())
        all_classes.append(probabilities.argmax(dim=1).cpu().numpy())
        all_expected.append((probabilities * intensity_tensor).sum(dim=1).cpu().numpy())
        all_attention.append(attention.cpu().numpy())
    return (
        np.concatenate(all_rows),
        np.concatenate(all_classes),
        np.concatenate(all_expected),
        np.concatenate(all_attention),
    )


def run_fold(
    fold: int,
    train_val_indices: np.ndarray,
    test_indices: np.ndarray,
    data: pd.DataFrame,
    raw_features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    contexts: np.ndarray,
    intensity_values: list[float],
    config: Config,
    device: torch.device,
    num_workers: int,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    fold_seed = config.seed + fold
    seed_everything(fold_seed)

    inner = GroupShuffleSplit(
        n_splits=1, test_size=config.validation_fraction, random_state=fold_seed,
    )
    inner_train, inner_val = next(inner.split(train_val_indices, groups=groups[train_val_indices]))
    train_indices = train_val_indices[inner_train]
    val_indices = train_val_indices[inner_val]
    transformed, _, _ = fit_preprocessor(raw_features, train_indices)

    train_loader = make_loader(transformed, labels, contexts, train_indices, config.batch_size, True, num_workers)
    val_loader = make_loader(transformed, labels, contexts, val_indices, config.batch_size, False, num_workers)
    test_loader = make_loader(transformed, labels, contexts, test_indices, config.batch_size, False, num_workers)

    counts = np.bincount(labels[train_indices], minlength=len(intensity_values)).astype(float)
    class_weights = np.zeros_like(counts)
    present = counts > 0
    class_weights[present] = len(train_indices) / (present.sum() * counts[present])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    model = AttentionTCN(
        n_features=len(FEATURES), n_classes=len(intensity_values),
        channels=config.channels, levels=config.levels,
        kernel_size=config.kernel_size, dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    gradient_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    stale_epochs = 0
    epochs_trained = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        for features, mask, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = model(features.to(device), mask.to(device))
                loss = criterion(logits, targets.to(device))
            gradient_scaler.scale(loss).backward()
            gradient_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            gradient_scaler.step(optimizer)
            gradient_scaler.update()

        model.eval()
        validation_loss = 0.0
        validation_count = 0
        with torch.no_grad():
            for features, mask, targets, _ in val_loader:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits, _ = model(features.to(device), mask.to(device))
                    batch_loss = criterion(logits, targets.to(device))
                validation_loss += batch_loss.item() * len(targets)
                validation_count += len(targets)
        validation_loss /= max(validation_count, 1)
        epochs_trained = epoch
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    intensity_tensor = torch.tensor(intensity_values, dtype=torch.float32, device=device)
    rows, predicted_classes, expected_intensity, attention = predict(model, test_loader, device, intensity_tensor)
    order = np.argsort(rows)
    rows, predicted_classes, expected_intensity, attention = (
        rows[order], predicted_classes[order], expected_intensity[order], attention[order]
    )
    actual = np.asarray(intensity_values)[labels[rows]]
    predicted = np.asarray(intensity_values)[predicted_classes]
    actual_classes = labels[rows]
    test_subject = str(groups[test_indices[0]])

    predictions = data.loc[rows, ["source_row", "dataset", "subject", "subject_id", "epoch", "laser_power"]].copy()
    predictions = predictions.rename(columns={"laser_power": "actual_laser_power"})
    predictions["predicted_laser_power"] = predicted
    predictions["expected_laser_power"] = expected_intensity
    predictions["attention_peak_lag"] = config.window - 1 - attention.argmax(axis=1)
    predictions["fold"] = fold

    metrics: dict[str, float | int | str] = {
        "fold": fold,
        "held_out_subject": test_subject,
        "n_test_intervals": len(rows),
        "epochs_trained": epochs_trained,
        "best_validation_loss": best_loss,
        "accuracy": float(accuracy_score(actual_classes, predicted_classes)),
        "balanced_accuracy": float(balanced_accuracy_score(actual_classes, predicted_classes)),
        "macro_f1": float(f1_score(actual_classes, predicted_classes, average="macro", zero_division=0)),
        "mae_joules": float(mean_absolute_error(actual, expected_intensity)),
        "rmse_joules": float(mean_squared_error(actual, expected_intensity) ** 0.5),
        "within_0.25_joules": float(np.mean(np.abs(actual - expected_intensity) <= 0.25)),
    }
    return predictions, metrics


def overall_metrics(predictions: pd.DataFrame) -> dict[str, float | int | str]:
    actual = predictions["actual_laser_power"].to_numpy()
    predicted = predictions["predicted_laser_power"].to_numpy()
    expected = predictions["expected_laser_power"].to_numpy()
    categories = sorted(set(actual).union(predicted))
    actual_classes = pd.Categorical(actual, categories=categories).codes
    predicted_classes = pd.Categorical(predicted, categories=categories).codes
    return {
        "cv": "LeaveOneGroupOut over dataset + subject",
        "n_intervals": int(len(predictions)),
        "n_completed_subjects": int(predictions["subject_id"].nunique()),
        "accuracy": float(accuracy_score(actual_classes, predicted_classes)),
        "balanced_accuracy": float(balanced_accuracy_score(actual_classes, predicted_classes)),
        "macro_f1": float(f1_score(actual_classes, predicted_classes, average="macro", zero_division=0)),
        "mae_joules": float(mean_absolute_error(actual, expected)),
        "rmse_joules": float(mean_squared_error(actual, expected) ** 0.5),
        "within_0.25_joules": float(np.mean(np.abs(actual - expected) <= 0.25)),
    }


def main() -> None:
    args = parse_args()
    config = Config(
        window=args.window, channels=args.channels, levels=args.levels,
        kernel_size=args.kernel_size, dropout=args.dropout,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction, seed=args.seed,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    selected_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if selected_device == "auto":
        selected_device = "cpu"
    device = torch.device(selected_device)

    data, labels, groups, intensity_values = load_data(args.csv, args.datasets)
    raw_features = data[FEATURES].to_numpy(dtype=np.float64)
    contexts = build_contexts(groups, config.window)
    splitter = LeaveOneGroupOut()
    n_folds = len(np.unique(groups))
    fold_end = args.fold_end or n_folds
    if not 1 <= args.fold_start <= fold_end <= n_folds:
        raise ValueError(f"Fold range must satisfy 1 <= start <= end <= {n_folds}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fragments = args.output_dir / "folds"
    fragments.mkdir(exist_ok=True)
    run_info = {
        "input_csv": str(args.csv.resolve()),
        "n_intervals": len(data),
        "n_dataset_specific_subjects": n_folds,
        "datasets": sorted(data["dataset"].unique().tolist()),
        "features": FEATURES,
        "intensity_values": intensity_values,
        "config": asdict(config),
        "device": str(device),
    }
    config_path = args.output_dir / "run_config.json"
    if config_path.exists() and any(fragments.glob("predictions_fold_*.csv")):
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != run_info:
            raise ValueError(
                "The output directory contains folds from a different configuration. "
                "Choose a new --output-dir to avoid mixing incompatible predictions."
            )
    config_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    print(json.dumps(run_info, indent=2))
    for fold, (train_val_indices, test_indices) in enumerate(splitter.split(raw_features, labels, groups), start=1):
        if fold < args.fold_start or fold > fold_end:
            continue
        prediction_path = fragments / f"predictions_fold_{fold:04d}.csv"
        metrics_path = fragments / f"metrics_fold_{fold:04d}.json"
        if prediction_path.exists() and metrics_path.exists() and not args.overwrite:
            print(f"Fold {fold}/{n_folds}: already complete; skipping")
            continue
        print(f"Fold {fold}/{n_folds}: held out {groups[test_indices[0]]} ({len(test_indices)} intervals)")
        predictions, metrics = run_fold(
            fold, train_val_indices, test_indices, data, raw_features, labels, groups,
            contexts, intensity_values, config, device, args.num_workers,
        )
        predictions.to_csv(prediction_path, index=False)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))

    prediction_files = sorted(fragments.glob("predictions_fold_*.csv"))
    metric_files = sorted(fragments.glob("metrics_fold_*.json"))
    if prediction_files:
        all_predictions = pd.concat((pd.read_csv(path) for path in prediction_files), ignore_index=True)
        all_predictions = all_predictions.sort_values("source_row")
        all_predictions.to_csv(args.output_dir / "oof_interval_predictions.csv", index=False)
        summary = overall_metrics(all_predictions)
        summary["is_complete_loso"] = len(prediction_files) == n_folds
        (args.output_dir / "overall_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    if metric_files:
        fold_metrics = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")) for path in metric_files)
        fold_metrics.sort_values("fold").to_csv(args.output_dir / "fold_metrics.csv", index=False)


if __name__ == "__main__":
    main()
