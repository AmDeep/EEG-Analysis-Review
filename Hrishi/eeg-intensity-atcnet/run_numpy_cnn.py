"""
Actually-runnable-today path: trains the pure-NumPy ATCNet-inspired CNN (models/numpy_cnn.py)
on Aditya's real features_combined.csv (29,516 trials, 678 subjects, target = laser_power in
Joules), and reports metrics comparable to Hrishi's existing baseline sweep
(model_comparison.md: Extra Trees MAE 0.4480 / RMSE 0.5955 / R2 0.3315).

No scikit-learn or torch dependency -- built to run in environments where pip/PyPI access is
blocked. Implements group-aware train/test split, median imputation, standardization, and
MAE/RMSE/R2/Pearson-r evaluation all in plain NumPy/pandas.

Usage:
    python run_numpy_cnn.py --csv "<path>/features_combined.csv" --include-rating
    python run_numpy_cnn.py --csv "<path>/features_combined.csv"          # EEG-only, no rating
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "models"))
from numpy_cnn import NumpyCNNRegressor  # noqa: E402

TARGET = "laser_power"
ID_COLS = {"dataset", "subject", "vertex_channel", "gamma_band_hz"}


def load_data(csv_path: str, include_rating: bool):
    df = pd.read_csv(csv_path)
    df["group"] = df["dataset"].astype(str) + "_" + df["subject"].astype(str)  # avoid cross-dataset subject-id collisions
    exclude = set(ID_COLS) | {TARGET, "group"}
    if not include_rating:
        exclude.add("rating")
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].to_numpy(dtype="float64")
    y = df[TARGET].to_numpy(dtype="float64")
    groups = df["group"].to_numpy()
    return X, y, groups, feature_cols


def group_split(groups, test_size=0.2, seed=42):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_size))
    test_groups = set(uniq[:n_test].tolist())
    is_test = np.array([g in test_groups for g in groups])
    return ~is_test, is_test


def median_impute_fit(X_train):
    return np.nanmedian(X_train, axis=0)


def median_impute_apply(X, medians):
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        col[np.isnan(col)] = medians[j]
    return X


def standardize_fit(X_train):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    yt_c = y_true - y_true.mean()
    yp_c = y_pred - y_pred.mean()
    pearson_r = np.sum(yt_c * yp_c) / (np.sqrt(np.sum(yt_c ** 2)) * np.sqrt(np.sum(yp_c ** 2)) + 1e-12)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "pearson_r": pearson_r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--include-rating", action="store_true", help="match teammates' baseline exactly (includes subjective pain rating as a feature)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    X, y, groups, feature_cols = load_data(args.csv, args.include_rating)
    print(f"Loaded {args.csv}")
    print(f"Rows: {len(X):,} | Features ({len(feature_cols)}): {feature_cols}")
    print(f"Unique dataset_subject groups: {len(np.unique(groups))}")

    train_mask, test_mask = group_split(groups, test_size=args.test_size, seed=args.seed)
    X_train_raw, X_test_raw = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"Train: {len(X_train_raw):,} rows | Test: {len(X_test_raw):,} rows")

    medians = median_impute_fit(X_train_raw)
    X_train = median_impute_apply(X_train_raw, medians)
    X_test = median_impute_apply(X_test_raw, medians)

    mean, std = standardize_fit(X_train)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = NumpyCNNRegressor(n_features=X_train.shape[1], seed=args.seed)
    print(f"\nTraining NumPy CNN for {args.epochs} epochs...")
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_metrics = metrics(y_train, train_pred)
    test_metrics = metrics(y_test, test_pred)

    elapsed = time.time() - t0
    print(f"\n=== Results (include_rating={args.include_rating}) ===")
    print(f"Train: MAE={train_metrics['MAE']:.4f}  RMSE={train_metrics['RMSE']:.4f}  R2={train_metrics['R2']:.4f}  r={train_metrics['pearson_r']:.4f}")
    print(f"Test:  MAE={test_metrics['MAE']:.4f}  RMSE={test_metrics['RMSE']:.4f}  R2={test_metrics['R2']:.4f}  r={test_metrics['pearson_r']:.4f}")
    print(f"(elapsed {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
