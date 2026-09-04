"""Shared, leakage-aware evaluation utilities for EEG laser-power models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight


DEFAULT_CSV = Path(r"C:\Users\patel\Documents\Codex\features_combined (3).csv")
RANDOM_SEED = 42
N_SPLITS = 5
TARGET = "laser_power"
GROUP_COLUMNS = ["dataset", "subject"]

# Only EEG features are used. Identifiers, epoch number, acquisition metadata,
# and the participant's behavioral rating are intentionally excluded.
EEG_FEATURES = [
    "n2_amp", "n2_lat", "p2_amp", "p2_lat", "n2p2_amp", "gamma_power",
    "alpha_erd_pct", "beta_erd_pct", "psd_delta", "psd_theta", "psd_alpha",
    "psd_beta", "psd_gamma", "plv_Fz-Cz", "plv_Cz-Pz", "plv_C3-C4",
    "plv_FCz-CPz", "perm_entropy", "spectral_entropy", "sample_entropy",
    "higuchi_fd", "dfa", "hjorth_mobility", "hjorth_complexity",
]


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help=f"Input CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results",
                        help="Directory for metrics and epoch-level out-of-fold predictions")
    return parser.parse_args()


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load data and return EEG features, power labels, and dataset-specific subject IDs."""
    data = pd.read_csv(csv_path)
    required = set(EEG_FEATURES + GROUP_COLUMNS + [TARGET])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    data = data.dropna(subset=[TARGET, *GROUP_COLUMNS]).copy()
    X = data.loc[:, EEG_FEATURES].apply(pd.to_numeric, errors="coerce")
    # A subject ID is only unique within a dataset, so the composite ID prevents
    # mixing epochs from the same study participant across train and test folds.
    groups = data["dataset"].astype(str) + "__" + data["subject"].astype(str)
    # Values represent discrete experimental set-points, not a continuous
    # regression target. Strings make that explicit to scikit-learn.
    y = data[TARGET].astype(float).map(lambda value: f"{value:.2f}")
    return X, y, groups


def evaluate_classifier(
    model: Any,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    output_dir: Path,
    balanced_sample_weights: bool = False,
) -> dict[str, float]:
    """Evaluate a classifier using five subject-held-out stratified folds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.sort(y.unique())
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    predictions = np.empty(len(y), dtype=object)
    fold_numbers = np.empty(len(y), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        fitted = clone(model)
        if balanced_sample_weights:
            weights = compute_sample_weight(class_weight="balanced", y=y.iloc[train_idx])
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx], model__sample_weight=weights)
        else:
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions[test_idx] = fitted.predict(X.iloc[test_idx])
        fold_numbers[test_idx] = fold

    results = {
        "model": model_name,
        "n_epochs": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_dataset_specific_subjects": int(groups.nunique()),
        "n_power_classes": int(len(labels)),
        "cv": "5-fold StratifiedGroupKFold; all epochs from one dataset-specific subject remain in one fold",
        "random_seed": RANDOM_SEED,
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, labels=labels, average="macro", zero_division=0)),
    }
    pd.DataFrame([results]).to_csv(output_dir / f"metrics_{model_name}.csv", index=False)
    with (output_dir / f"metrics_{model_name}.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    oof = pd.DataFrame({"actual_laser_power": y, "predicted_laser_power": predictions,
                        "fold": fold_numbers, "dataset_specific_subject": groups})
    oof.to_csv(output_dir / f"oof_predictions_{model_name}.csv", index=False)
    matrix = pd.DataFrame(confusion_matrix(y, predictions, labels=labels), index=labels, columns=labels)
    matrix.index.name = "actual_laser_power"
    matrix.columns.name = "predicted_laser_power"
    matrix.to_csv(output_dir / f"confusion_matrix_{model_name}.csv")
    print(json.dumps(results, indent=2))
    return results
