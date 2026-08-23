"""Leakage-safe LOSO ablation of the expanded EEG biomarker features.

This script compares the validated 135-feature matrix with the 165 expanded
features from Notebook 05. Every operation that can learn from the data
(non-finite-value replacement, variance filtering, scaling, ANOVA selection,
and LDA fitting) is performed independently inside each held-out-subject
fold. The historical 0.850 value in Notebook 04 used global feature
selection, so it is retained only as a reference and is not used for fitting.

Run from the repository root:

    python notebooks/06_biomarker_loso_ablation.py

Outputs are written to notebooks/data/preprocessed/:
    biomarker_loso_ablation_results.csv
    biomarker_loso_ablation_folds.csv
    biomarker_loso_ablation_config.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "1.0"
HISTORICAL_BASELINE_AUC = 0.850
VARIANCE_THRESHOLD = 1e-8
MAX_SELECTED_FEATURES = 60


def expanded_family(feature_name: str) -> str:
    """Assign an expanded feature to exactly one requested family."""
    if "wavelet_" in feature_name:
        return "wavelet"
    if "mfcc" in feature_name:
        return "mfcc"
    if "arburg" in feature_name:
        return "ar"
    if "-" in feature_name:
        return "connectivity"
    spectral_tokens = (
        "_power",
        "_relative",
        "_aperiodic_slope",
        "_spectral_centroid",
        "_spectral_edge_50",
        "_spectral_edge_95",
        "_alpha_peak_frequency",
    )
    if any(token in feature_name for token in spectral_tokens):
        return "spectral"
    return "other_expanded"


def safe_matrix(values: np.ndarray) -> np.ndarray:
    """Make input finite without learning replacement values from any fold."""
    return np.nan_to_num(
        np.asarray(values, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def select_and_fit_lda(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit all preprocessing and LDA parameters on training rows only."""
    x_train = safe_matrix(x_train)
    x_test = safe_matrix(x_test)

    # The variance threshold is fitted on training subjects only. A constant
    # fallback keeps the fold explicit if an unusually small feature family
    # has no non-constant columns.
    variance = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    try:
        x_train_var = variance.fit_transform(x_train)
        x_test_var = variance.transform(x_test)
    except ValueError:
        x_train_var = x_train[:, :1]
        x_test_var = x_test[:, :1]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_var)
    x_test_scaled = scaler.transform(x_test_var)

    n_selected = min(MAX_SELECTED_FEATURES, x_train_scaled.shape[1])
    selector = SelectKBest(score_func=f_classif, k=n_selected)
    x_train_selected = selector.fit_transform(x_train_scaled, y_train)
    x_test_selected = selector.transform(x_test_scaled)

    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(x_train_selected, y_train)
    return model.predict(x_test_selected), model.predict_proba(x_test_selected)[:, 1], n_selected


def evaluate_feature_set(
    name: str,
    indices: np.ndarray,
    baseline_x: np.ndarray,
    expanded_x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    baseline_feature_count: int,
    expanded_feature_count: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate one feature set using one LOSO fit per subject."""
    x = np.concatenate(
        [baseline_x, expanded_x[:, indices]],
        axis=1,
    )
    logo = LeaveOneGroupOut()
    fold_rows: list[dict[str, object]] = []
    all_y, all_scores = [], []
    fold_metrics = {key: [] for key in ("accuracy", "balanced_accuracy", "auc", "f1", "sensitivity", "specificity")}
    selected_counts = []

    for fold_number, (train, test) in enumerate(logo.split(x, y, groups), start=1):
        y_test = y[test]
        predictions, scores, n_selected = select_and_fit_lda(x[train], y[train], x[test])
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "auc": roc_auc_score(y_test, scores) if len(np.unique(y_test)) > 1 else 0.5,
            "f1": f1_score(y_test, predictions, pos_label=1, zero_division=0),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
        }
        for key, value in metrics.items():
            fold_metrics[key].append(float(value))
        selected_counts.append(n_selected)
        all_y.extend(y_test.tolist())
        all_scores.extend(scores.tolist())
        fold_rows.append(
            {
                "feature_set": name,
                "fold": fold_number,
                "held_out_subject": str(groups[test][0]),
                "train_epochs": len(train),
                "test_epochs": len(test),
                "input_features": int(x.shape[1]),
                "expanded_features_added": int(len(indices)),
                "selected_features": int(n_selected),
                **metrics,
            }
        )

    aucs = np.asarray(fold_metrics["auc"], dtype=float)
    result = {
        "feature_set": name,
        "baseline_features": int(baseline_feature_count),
        "expanded_features_added": int(len(indices)),
        "expanded_features_total": int(expanded_feature_count),
        "input_features": int(x.shape[1]),
        "folds": len(fold_rows),
        "accuracy": float(np.mean(fold_metrics["accuracy"])),
        "balanced_accuracy": float(np.mean(fold_metrics["balanced_accuracy"])),
        # This is the same macro-over-subject AUC convention as Notebook 04.
        "auc_macro_subject": float(np.mean(aucs)),
        "auc_std_subject": float(np.std(aucs)),
        "auc_pooled_epochs": float(roc_auc_score(all_y, all_scores)),
        "f1": float(np.mean(fold_metrics["f1"])),
        "sensitivity": float(np.mean(fold_metrics["sensitivity"])),
        "specificity": float(np.mean(fold_metrics["specificity"])),
        "selected_features_mean": float(np.mean(selected_counts)),
        "selected_features_min": int(np.min(selected_counts)),
        "selected_features_max": int(np.max(selected_counts)),
    }
    return result, fold_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_auc_comparison(
    fold_results: pd.DataFrame,
    feature_set: str,
) -> tuple[int, int, float]:
    """Compare a feature set with the matched held-out-subject baseline AUCs."""
    baseline = (
        fold_results.loc[fold_results["feature_set"].eq("baseline_135")]
        .set_index("held_out_subject")["auc"]
        .sort_index()
    )
    candidate = (
        fold_results.loc[fold_results["feature_set"].eq(feature_set)]
        .set_index("held_out_subject")["auc"]
        .sort_index()
    )
    differences = (candidate - baseline).to_numpy(dtype=float)
    improved = int(np.sum(differences > 0))
    worsened = int(np.sum(differences < 0))
    if feature_set == "baseline_135" or np.allclose(differences, 0.0):
        return improved, worsened, 1.0
    # A paired, one-sided test asks whether the candidate's held-out AUC is
    # greater than baseline across the same 26 subjects.
    p_value = float(
        stats.wilcoxon(
            differences,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )
    return improved, worsened, p_value


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "notebooks" / "data" / "preprocessed"
    baseline_path = data_dir / "features_xy.npz"
    expanded_path = data_dir / "biomarker_exploration_features.npz"
    results_path = data_dir / "biomarker_loso_ablation_results.csv"
    folds_path = data_dir / "biomarker_loso_ablation_folds.csv"
    config_path = data_dir / "biomarker_loso_ablation_config.json"

    baseline = np.load(baseline_path, allow_pickle=True)
    expanded = np.load(expanded_path, allow_pickle=True)
    baseline_x = safe_matrix(baseline["X"])
    expanded_x = safe_matrix(expanded["X"])
    y = np.asarray(baseline["y"], dtype=int)
    groups = np.asarray(baseline["subjects"], dtype=str)
    baseline_names = [str(value) for value in baseline["feature_names"]]
    expanded_names = [str(value) for value in expanded["feature_names"]]

    if not np.array_equal(y, expanded["y"]) or not np.array_equal(groups, expanded["subjects"].astype(str)):
        raise ValueError("Baseline and expanded matrices do not have identical row labels/groups.")
    if len(np.unique(groups)) < 2:
        raise ValueError("LOSO requires at least two subjects.")

    family_to_indices: dict[str, np.ndarray] = {}
    for family in ("wavelet", "mfcc", "ar", "spectral", "connectivity", "other_expanded"):
        family_to_indices[family] = np.asarray(
            [i for i, name in enumerate(expanded_names) if expanded_family(name) == family],
            dtype=int,
        )
    requested_families = ("wavelet", "mfcc", "ar", "spectral", "connectivity")
    if any(len(family_to_indices[family]) == 0 for family in requested_families):
        raise ValueError(f"Missing requested expanded feature family: {family_to_indices}")

    all_expanded_indices = np.arange(len(expanded_names), dtype=int)
    family_sets: list[tuple[str, np.ndarray]] = [
        ("baseline_135", np.asarray([], dtype=int)),
        *[(f"{family}_only", family_to_indices[family]) for family in requested_families],
        ("all_expanded_only", all_expanded_indices),
        *[
            (f"baseline_plus_{family}", family_to_indices[family])
            for family in requested_families
        ],
        ("baseline_plus_all_expanded", all_expanded_indices),
    ]

    print(f"Loaded baseline matrix: {baseline_x.shape}")
    print(f"Loaded expanded matrix: {expanded_x.shape}")
    print(f"Subjects: {len(np.unique(groups))}; labels: {np.bincount(y).tolist()}")
    print("Expanded family counts:", {key: len(value) for key, value in family_to_indices.items()})
    print("\nRunning fold-local LOSO ablations...")

    result_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for feature_set, indices in family_sets:
        result, folds = evaluate_feature_set(
            feature_set,
            indices,
            baseline_x if feature_set == "baseline_135" or "baseline_plus" in feature_set else np.empty((len(y), 0)),
            expanded_x,
            y,
            groups,
            len(baseline_names) if feature_set == "baseline_135" or "baseline_plus" in feature_set else 0,
            len(expanded_names),
        )
        result_rows.append(result)
        fold_rows.extend(folds)
        print(
            f"  {feature_set:28s} "
            f"AUC={result['auc_macro_subject']:.4f} "
            f"ACC={result['accuracy']:.4f} "
            f"selected={result['selected_features_mean']:.1f}"
        )

    results = pd.DataFrame(result_rows)
    folds = pd.DataFrame(fold_rows)
    baseline_auc = float(results.loc[results["feature_set"].eq("baseline_135"), "auc_macro_subject"].iloc[0])
    results["delta_auc_vs_fold_local_baseline"] = results["auc_macro_subject"] - baseline_auc
    results["delta_auc_vs_historical_0850"] = results["auc_macro_subject"] - HISTORICAL_BASELINE_AUC
    results["improves_fold_local_baseline"] = results["delta_auc_vs_fold_local_baseline"] > 0
    results["improves_historical_0850"] = results["auc_macro_subject"] > HISTORICAL_BASELINE_AUC
    paired = [
        paired_auc_comparison(folds, feature_set)
        for feature_set in results["feature_set"]
    ]
    results["subjects_auc_better_than_baseline"] = [value[0] for value in paired]
    results["subjects_auc_worse_than_baseline"] = [value[1] for value in paired]
    results["paired_wilcoxon_p_auc_greater_than_baseline"] = [value[2] for value in paired]
    results.to_csv(results_path, index=False, float_format="%.8f")
    folds.to_csv(folds_path, index=False, float_format="%.8f")

    config = {
        "script_version": SCRIPT_VERSION,
        "historical_baseline_auc": HISTORICAL_BASELINE_AUC,
        "normalization": "StandardScaler fit on training rows in each LOSO fold; no test-subject statistics used",
        "historical_comparison_note": (
            "Notebook 04's 0.850 used global feature selection and per-subject "
            "z-normalization, including held-out-subject statistics. It is a "
            "legacy reference only; baseline_135 is the fair inferential comparator."
        ),
        "variance_filter": f"VarianceThreshold({VARIANCE_THRESHOLD}) fit on training rows in each fold",
        "feature_selection": f"SelectKBest(f_classif, k=min({MAX_SELECTED_FEATURES}, training-fold non-constant features))",
        "classifier": "LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')",
        "cv": "LeaveOneGroupOut with subject as group; macro-subject AUC",
        "paired_comparison": "One-sided Wilcoxon signed-rank test on matched subject-fold AUCs versus baseline_135",
        "baseline_input": str(baseline_path.relative_to(project_root)),
        "expanded_input": str(expanded_path.relative_to(project_root)),
        "baseline_input_sha256": sha256(baseline_path),
        "expanded_input_sha256": sha256(expanded_path),
        "baseline_feature_count": len(baseline_names),
        "expanded_feature_count": len(expanded_names),
        "expanded_family_counts": {key: int(len(value)) for key, value in family_to_indices.items()},
        "package_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "subjects": sorted(np.unique(groups).tolist()),
        "feature_sets": [name for name, _ in family_sets],
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    display_columns = [
        "feature_set",
        "input_features",
        "expanded_features_added",
        "accuracy",
        "balanced_accuracy",
        "auc_macro_subject",
        "auc_std_subject",
        "auc_pooled_epochs",
        "delta_auc_vs_fold_local_baseline",
        "delta_auc_vs_historical_0850",
        "subjects_auc_better_than_baseline",
        "subjects_auc_worse_than_baseline",
        "paired_wilcoxon_p_auc_greater_than_baseline",
        "improves_historical_0850",
    ]
    print("\nResults:")
    print(results[display_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote {results_path}")
    print(f"Wrote {folds_path}")
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    sys.exit(main())