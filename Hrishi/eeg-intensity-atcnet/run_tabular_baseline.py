"""
Quick path: train TabularATCNetRegressor directly on Aditya's features-combined CSV,
no raw-EEG download needed. Auto-detects channel/feature columns and the label column
by name so it works without knowing the exact schema in advance -- override with
--label-col if the auto-detection guesses wrong (it prints what it found).

Usage:
    python run_tabular_baseline.py --csv /path/to/features_combined.csv --label-col laser_power
    python run_tabular_baseline.py --csv /path/to/features_combined.csv --label-col laser_power --drop-rating
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.atcnet import TabularATCNetRegressor

LABEL_CANDIDATES = [
    "laser_power", "stimulus_intensity", "stim_intensity", "intensity", "energy_j",
    "laser_intensity", "target", "label", "y", "intensity_level", "condition",
]
SUBJECT_CANDIDATES = ["subject", "subject_id", "participant", "participant_id", "subj"]
KNOWN_CHANNELS = ["C3", "CZ", "C4", "FZ", "PZ", "F3", "F4", "P3", "P4"]
# columns that identify a row rather than describe it (dropped from the feature matrix, not
# just from channel-grouping) -- e.g. Aditya's features_combined.csv has 'dataset',
# 'vertex_channel', 'gamma_band_hz' alongside the real numeric features.
NON_FEATURE_COLUMNS = ["dataset", "vertex_channel", "gamma_band_hz", "sfreq", "epoch"]


def detect_label_column(df: pd.DataFrame, override: str | None) -> str:
    if override:
        return override
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in LABEL_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    raise ValueError(
        f"Couldn't auto-detect a label column among {list(df.columns)}. "
        f"Pass --label-col <name> explicitly."
    )


def detect_subject_column(df: pd.DataFrame) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in SUBJECT_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def detect_channel_feature_columns(df: pd.DataFrame, label_col: str, subject_col: str | None):
    """
    Group feature columns like 'CZ_higuchi', 'C3_alpha_power' by channel prefix if the CSV
    uses that naming convention. If it doesn't (e.g. Aditya's features_combined.csv, which has
    flat names like 'n2_amp', 'psd_delta', 'plv_Fz-Cz' plus a separate 'vertex_channel' column
    identifying which single channel the row's ERP/entropy features came from), fall back to
    treating every numeric feature as one pseudo-channel -- still valid input for
    TabularATCNetRegressor with n_channels=1, just without the per-channel conv structure.
    """
    exclude = {label_col} | set(NON_FEATURE_COLUMNS)
    if subject_col:
        exclude.add(subject_col)
    candidate_cols = [c for c in df.columns if c not in exclude]
    # keep only columns that are actually numeric (drops stray string/id columns we didn't anticipate)
    feature_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c]) or
                    pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9]
    dropped = [c for c in candidate_cols if c not in feature_cols]
    if dropped:
        print(f"  [info] dropped non-numeric columns: {dropped}")

    channel_groups: dict[str, list[str]] = {}
    leftover = []
    for c in feature_cols:
        m = re.match(r"([A-Za-z0-9]+)_(.+)", c)
        prefix = m.group(1).upper() if m else None
        if prefix and prefix in KNOWN_CHANNELS:
            channel_groups.setdefault(prefix, []).append(c)
        else:
            leftover.append(c)

    if not channel_groups:
        print(f"  [info] no 'CH_feature' naming convention detected; using all {len(feature_cols)} "
              f"features as a single pseudo-channel (n_channels=1): {feature_cols}")
        return ["ALL"], [feature_cols]

    if leftover:
        print(f"  [info] {len(leftover)} columns didn't match a known channel prefix and are dropped: {leftover[:10]}{'...' if len(leftover) > 10 else ''}")

    n_per_channel = min(len(v) for v in channel_groups.values())
    if len({len(v) for v in channel_groups.values()}) > 1:
        print(f"  [warn] channels have different feature counts {[(k, len(v)) for k, v in channel_groups.items()]}; truncating to {n_per_channel} each")

    channels = sorted(channel_groups)
    matrix_cols = [channel_groups[ch][:n_per_channel] for ch in channels]
    return channels, matrix_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--drop-rating", action="store_true", help="exclude the subjective 'rating' column from features (the more honest EEG-only setup)")
    ap.add_argument("--epochs", type=int, default=50, help="the NumPy-CNN run (RESULTS.md) started overfitting past ~40 epochs on this data, so 50 is a safer default than the original 150")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {args.csv}: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:", list(df.columns))

    label_col = detect_label_column(df, args.label_col)
    subject_col = detect_subject_column(df)
    if subject_col and "dataset" in df.columns:
        # avoid silently merging different real subjects that share a "sub-001"-style id
        # across different sub-experiments/datasets
        df["_group"] = df["dataset"].astype(str) + "_" + df[subject_col].astype(str)
        subject_col = "_group"
        print("  [info] combined 'dataset' + subject id into a single group column to avoid "
              "cross-dataset subject-id collisions")
    print(f"Using label column: '{label_col}'" + (f", subject column: '{subject_col}'" if subject_col else " (no subject column found -- using a random split, not grouped)"))

    if args.drop_rating and "rating" in df.columns:
        df = df.drop(columns=["rating"])
        print("  [info] dropped 'rating' column per --drop-rating (EEG-only features)")

    channels, matrix_cols = detect_channel_feature_columns(df, label_col, subject_col)
    print(f"Detected channels: {channels}, {len(matrix_cols[0])} features each")

    df = df.dropna(subset=[label_col])
    y = df[label_col].astype(float).values

    X = np.stack([df[cols].astype(float).values for cols in matrix_cols], axis=1)  # (N, n_channels, n_features)
    X = np.nan_to_num(X, nan=0.0)

    if subject_col:
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=0)
        train_idx, test_idx = next(splitter.split(X, y, df[subject_col].values))
    else:
        train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=args.test_size, random_state=0)

    # standardize features (fit on train only) and normalize label to [0,1] (fit on train only)
    n, c, f = X.shape
    scaler = StandardScaler().fit(X[train_idx].reshape(len(train_idx) * c, f))
    X_scaled = scaler.transform(X.reshape(n * c, f)).reshape(n, c, f)

    y_min, y_max = y[train_idx].min(), y[train_idx].max()
    y_norm = (y - y_min) / max(1e-8, (y_max - y_min))

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU -- meaningfully faster than CPU for this
    else:
        device = "cpu"
    print(f"device: {device}  |  train rows: {len(train_idx):,}  |  batches/epoch: {-(-len(train_idx)//args.batch_size):,}")
    model = TabularATCNetRegressor(n_channels=c, n_features=f).to(device)

    train_ds = TensorDataset(torch.tensor(X_scaled[train_idx], dtype=torch.float32), torch.tensor(y_norm[train_idx], dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_scaled[test_idx], dtype=torch.float32), torch.tensor(y_norm[test_idx], dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        elapsed = time.time() - t0
        print(f"epoch {epoch:3d}: train_loss={epoch_loss / n_batches:.4f}  ({elapsed:.1f}s/epoch, "
              f"~{elapsed * (args.epochs - epoch - 1) / 60:.1f} min left)", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    preds = np.concatenate(preds)
    y_test = y_norm[test_idx]

    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)
    r, p = pearsonr(y_test, preds)
    print(f"\nTest metrics (n={len(y_test)}): MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  Pearson r={r:.4f} (p={p:.2e})")


if __name__ == "__main__":
    main()
