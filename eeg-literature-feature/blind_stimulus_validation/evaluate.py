"""Frozen models, true cohort holdouts, participant splits and negative controls."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.linalg import eigh
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FAMILIES, feature_set

HERE = Path(__file__).resolve().parent
SEED = 20261009


def model(kind, task):
    if task == "regression":
        estimator = {
            "dummy": lambda: DummyRegressor(strategy="mean"),
            "ridge": lambda: Ridge(alpha=10),
            "extra_trees": lambda: ExtraTreesRegressor(
                n_estimators=200, max_depth=8, min_samples_leaf=8,
                max_features=0.7, random_state=SEED, n_jobs=2)
        }[kind]()
    else:
        estimator = {
            "dummy": lambda: DummyClassifier(strategy="most_frequent"),
            "logistic": lambda: LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=3000, random_state=SEED),
            "extra_trees": lambda: ExtraTreesClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=8,
                max_features=0.7, class_weight="balanced", random_state=SEED, n_jobs=2)
        }[kind]()
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), estimator)


def csp(train_cov, test_cov, binary_y):
    """Equivalent two-filter CSP, using only TRAINING class covariance means."""
    train = train_cov / np.maximum(np.trace(train_cov, axis1=1, axis2=2)[:, None, None], 1e-12)
    test = test_cov / np.maximum(np.trace(test_cov, axis1=1, axis2=2)[:, None, None], 1e-12)
    if len(np.unique(binary_y)) < 2:
        raise ValueError("CSP needs both training stimulus classes")
    a, b = train[binary_y == 0].mean(axis=0), train[binary_y == 1].mean(axis=0)
    eye = np.eye(3) / 3
    a, b = 0.9 * a + 0.1 * eye, 0.9 * b + 0.1 * eye
    _, vectors = eigh(a, a + b)
    weights = vectors[:, [0, -1]]

    def transform(cov):
        variance = np.einsum("ai,nab,bi->ni", weights, cov, weights)
        return np.log(np.maximum(variance, 1e-12))

    return transform(train), transform(test)


def matrices(X, covariance, names, train_idx, test_idx, which, binary_y, omit=None):
    mask = feature_set(names, which, omit)
    train, test = X[train_idx][:, mask], X[test_idx][:, mask]
    if which == "all_compatible" and omit != "csp":
        a, b = csp(covariance[train_idx], covariance[test_idx], binary_y)
        train, test = np.column_stack([train, a]), np.column_stack([test, b])
    return train, test


def scores(y, prediction, task, probability=None):
    if task == "regression":
        levels = np.array([42., 46., 48.])
        rounded = levels[np.argmin(np.abs(prediction[:, None] - levels), axis=1)]
        return {
            "R2": r2_score(y, prediction), "MAE": mean_absolute_error(y, prediction),
            "RMSE": np.sqrt(mean_squared_error(y, prediction)),
            "native_level_accuracy": accuracy_score(y, rounded),
            "native_level_balanced_accuracy": balanced_accuracy_score(y, rounded)
        }
    return {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "ROC_AUC": roc_auc_score(y, probability) if len(np.unique(y)) == 2 else np.nan,
        "macro_F1": f1_score(y, prediction, average="macro", zero_division=0)
    }


def bootstrap(frame, task):
    rng = np.random.default_rng(SEED)
    groups = frame.subject.unique()
    indices = {s: np.flatnonzero(frame.subject.to_numpy() == s) for s in groups}
    y = frame.true_target.to_numpy()
    p = frame.prediction.to_numpy()
    prob = frame.probability.to_numpy()
    collected = {}
    for _ in range(1000):
        idx = np.concatenate([indices[s] for s in rng.choice(groups, len(groups), replace=True)])
        # All chosen subjects should have both classes; disclose any undefined replicate.
        if task != "regression" and len(np.unique(y[idx])) < 2:
            continue
        values = scores(y[idx], p[idx], task, prob[idx])
        for k, v in values.items():
            if np.isfinite(v):
                collected.setdefault(k, []).append(v)
    return {f"{k}_CI_{tail}": float(np.quantile(v, q))
            for k, v in collected.items() for tail, q in [("low", 0.025), ("high", 0.975)]}


def prediction_rows(meta, test_idx, estimator, test, task, tags, fold):
    y = meta.target_native.to_numpy() if task == "regression" else meta.binary_target.to_numpy()
    prediction = estimator.predict(test)
    prob = (estimator.predict_proba(test)[:, 1] if task != "regression"
            else np.full(len(test_idx), np.nan))
    return pd.DataFrame({
        **tags, "fold": fold, "subject": meta.subject.to_numpy()[test_idx],
        "source_trial": meta.source_trial.to_numpy()[test_idx],
        "true_target": y[test_idx], "prediction": prediction, "probability": prob
    })


def freeze_code():
    files = ["protocol.json", "protocol_availability_amendment.json", "acquire.py", "readers.py", "features.py",
             "build_features.py", "evaluate.py"]
    digests = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in files}
    path = HERE / "analysis_frozen.json"
    if path.exists():
        before = json.loads(path.read_text())
        assert before["file_sha256"] == digests, "Analysis changed after predictive evaluation freeze"
    else:
        path.write_text(json.dumps({
            "UTC_before_first_model_fit": datetime.now(timezone.utc).isoformat(),
            "file_sha256": digests,
            "implementation_detail": "Fourth-order causal 1-40 Hz filter, fixed -10..+30 ms trigger interpolation for every class, resample 500 to250Hz; source shock filtering remains irreversible.",
            "native_level_accuracy_definition": "Regressor predictions rounded to nearest prespecified 42/46/48 C; no threshold optimized on data.",
            "retrospective": True,
            "not_public_preregistration": True
        }, indent=2))


def main():
    freeze_code()
    data = np.load(HERE / "features.npz")
    X, covariance, names = data["X"], data["covariance"], data["names"]
    meta = pd.read_csv(HERE / "trial_metadata.csv")
    if len(meta) != len(X):
        raise ValueError("Features / target rows not aligned")
    for dataset, part in meta.groupby("dataset"):
        if part.subject.nunique() < 8 or part.binary_target.nunique() != 2:
            raise ValueError(f"Insufficient usable participants/classes in {dataset}")
    y_binary = meta.binary_target.to_numpy()
    datasets = sorted(meta.dataset.unique())
    feature_sets = ["power_only", "all_compatible", "prestimulus_only"]
    predictions = []
    split_audit = []

    def run(train_idx, test_idx, evaluation, train_key, test_key, task, fold):
        train_subjects = set(meta.subject.iloc[train_idx])
        test_subjects = set(meta.subject.iloc[test_idx])
        assert not train_subjects.intersection(test_subjects), "Participant leakage"
        if evaluation == "cross_dataset":
            assert set(meta.dataset.iloc[train_idx]).isdisjoint(set(meta.dataset.iloc[test_idx]))
        split_audit.append({
            "evaluation": evaluation, "train_dataset": train_key, "test_dataset": test_key,
            "fold": fold, "task": task, "train_subjects": sorted(train_subjects),
            "test_subjects": sorted(test_subjects), "subject_overlap": [],
            "training_rows": len(train_idx), "testing_rows": len(test_idx)
        })
        target = meta.target_native.to_numpy() if task == "regression" else y_binary
        kinds = ["dummy", "ridge", "extra_trees"] if task == "regression" else ["dummy", "logistic", "extra_trees"]
        for which in feature_sets:
            train, test = matrices(X, covariance, names, train_idx, test_idx,
                                   which, y_binary[train_idx])
            for kind in kinds:
                estimator = model(kind, task).fit(train, target[train_idx])
                tags = {"evaluation": evaluation, "train_dataset": train_key,
                        "test_dataset": test_key, "task": task, "feature_set": which,
                        "model": kind, "units": "degrees Celsius" if task == "regression" else "source-defined low/higher contrast 0/1"}
                predictions.append(prediction_rows(meta, test_idx, estimator, test, task, tags, fold))
        print(evaluation, train_key, "->", test_key, task, "fold", fold, flush=True)

    if len(datasets) >= 2:
        for heldout in datasets:
            test_idx = np.flatnonzero(meta.dataset.to_numpy() == heldout)
            train_idx = np.flatnonzero(meta.dataset.to_numpy() != heldout)
            run(train_idx, test_idx, "cross_dataset", next(d for d in datasets if d != heldout),
                heldout, "classification", 0)
    else:
        print("CROSS-DATASET VALIDATION BLOCKED: only one eligible independently sourced cohort. No transfer scores produced.", flush=True)
    for dataset in datasets:
        idx = np.flatnonzero(meta.dataset.to_numpy() == dataset)
        part = meta.iloc[idx]
        for fold, (train, test) in enumerate(GroupKFold(n_splits=4).split(
                np.zeros(len(idx)), groups=part.subject.to_numpy())):
            run(idx[train], idx[test], "subject_heldout", dataset, dataset, "classification", fold)
            if dataset.startswith("heat"):
                run(idx[train], idx[test], "subject_heldout", dataset, dataset, "regression", fold)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(HERE / "heldout_predictions.csv", index=False)
    (HERE / "split_leakage_audit.json").write_text(json.dumps(split_audit, indent=2))
    keys = ["evaluation", "train_dataset", "test_dataset", "task", "feature_set", "model", "units"]
    results, participant_results = [], []
    for values, part in all_predictions.groupby(keys):
        tags = dict(zip(keys, values))
        task = tags["task"]
        measured = scores(part.true_target.to_numpy(), part.prediction.to_numpy(), task,
                          part.probability.to_numpy())
        results.append({**tags, "test_subjects": part.subject.nunique(), "test_trials": len(part),
                        **measured, **bootstrap(part, task)})
        for subject, sp in part.groupby("subject"):
            participant_results.append({**tags, "subject": subject, "test_trials": len(sp),
                                        **scores(sp.true_target.to_numpy(), sp.prediction.to_numpy(),
                                                 task, sp.probability.to_numpy())})
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(HERE / "accuracy_results.csv", index=False)
    pd.DataFrame(participant_results).to_csv(HERE / "participant_accuracy_results.csv", index=False)
    print("Measured holdout scores and clustered intervals saved", flush=True)
    gain_rows = []
    gain_evaluation = "cross_dataset" if len(datasets) >= 2 else "subject_heldout"
    cross = all_predictions[(all_predictions.evaluation == gain_evaluation) &
                            (all_predictions.model == "logistic")]
    for heldout in datasets:
        full = cross[(cross.test_dataset == heldout) &
                     (cross.feature_set == "all_compatible")].sort_values(["subject", "source_trial"])
        power = cross[(cross.test_dataset == heldout) &
                      (cross.feature_set == "power_only")].sort_values(["subject", "source_trial"])
        assert full[["subject", "source_trial"]].to_numpy().tolist() == power[["subject", "source_trial"]].to_numpy().tolist()
        full, power = full.reset_index(drop=True), power.reset_index(drop=True)
        groups = full.subject.unique()
        lookup = {s: np.flatnonzero(full.subject.to_numpy() == s) for s in groups}
        y, fp, pp = full.true_target.to_numpy(), full.prediction.to_numpy(), power.prediction.to_numpy()
        gains, rng = [], np.random.default_rng(SEED)
        for _ in range(1000):
            ix = np.concatenate([lookup[s] for s in rng.choice(groups, len(groups), replace=True)])
            gains.append(balanced_accuracy_score(y[ix], fp[ix]) - balanced_accuracy_score(y[ix], pp[ix]))
        gain_rows.append({"evaluation": gain_evaluation, "test_dataset": heldout,
                          "balanced_accuracy_gain_full_minus_power": balanced_accuracy_score(y, fp) - balanced_accuracy_score(y, pp),
                          "gain_CI_low": np.quantile(gains, .025), "gain_CI_high": np.quantile(gains, .975)})
    pd.DataFrame(gain_rows).to_csv(HERE / "feature_set_gains.csv", index=False)

    # Prespecified family ablations, no selection/tuning on the resulting scores.
    ablations, nulls = [], []
    for heldout in datasets:
        if len(datasets) >= 2:
            folds = [(np.flatnonzero(meta.dataset.to_numpy() != heldout),
                      np.flatnonzero(meta.dataset.to_numpy() == heldout))]
        else:
            indices = np.flatnonzero(meta.dataset.to_numpy() == heldout)
            folds = [(indices[a], indices[b]) for a, b in GroupKFold(n_splits=4).split(
                indices, groups=meta.subject.to_numpy()[indices])]
        observed = cross[(cross.test_dataset == heldout) & (cross.feature_set == "all_compatible")]
        observed_score = balanced_accuracy_score(observed.true_target, observed.prediction)
        for omit, label in FAMILIES.items():
            truths, predictions_removed = [], []
            for train_idx, test_idx in folds:
                a, b = matrices(X, covariance, names, train_idx, test_idx,
                                "all_compatible", y_binary[train_idx], omit=omit)
                estimator = model("logistic", "classification").fit(a, y_binary[train_idx])
                truths.extend(y_binary[test_idx])
                predictions_removed.extend(estimator.predict(b))
            removed_score = balanced_accuracy_score(truths, predictions_removed)
            ablations.append({"evaluation": gain_evaluation, "test_dataset": heldout, "removed_family": label,
                              "balanced_accuracy_without_family": removed_score,
                              "balanced_accuracy_full": observed_score,
                              "full_minus_without_family": observed_score - removed_score,
                              "interpretation": "Positive = family helps at this holdout; exploratory, correlated families, no independent selection validation"})
        rng = np.random.default_rng(SEED)
        for replicate in range(20):
            truths, predictions_permuted = [], []
            for train_idx, test_idx in folds:
                groups = meta.subject.to_numpy()[train_idx]
                permuted = y_binary[train_idx].copy()
                for subject in np.unique(groups):
                    local = np.flatnonzero(groups == subject)
                    permuted[local] = rng.permutation(permuted[local])
                a, b = matrices(X, covariance, names, train_idx, test_idx,
                                "all_compatible", permuted)
                estimator = model("logistic", "classification").fit(a, permuted)
                truths.extend(y_binary[test_idx])
                predictions_permuted.extend(estimator.predict(b))
            value = balanced_accuracy_score(truths, predictions_permuted)
            nulls.append({"evaluation": gain_evaluation, "test_dataset": heldout, "permutation": replicate,
                          "permuted_balanced_accuracy": value, "observed_balanced_accuracy": observed_score})
        print("Family ablations and 20 label permutations:", heldout, flush=True)
    pd.DataFrame(ablations).to_csv(HERE / "family_ablation_results.csv", index=False)
    null_frame = pd.DataFrame(nulls)
    null_frame.to_csv(HERE / "permutation_controls.csv", index=False)
    null_summary = []
    for heldout, part in null_frame.groupby("test_dataset"):
        null_summary.append({"test_dataset": heldout, "permutations": len(part),
                             "observed_balanced_accuracy": part.observed_balanced_accuracy.iloc[0],
                             "mean_permuted_balanced_accuracy": part.permuted_balanced_accuracy.mean(),
                             "empirical_one_sided_p": (1 + (part.permuted_balanced_accuracy >= part.observed_balanced_accuracy).sum()) / (1 + len(part)),
                             "limitation": "20 permutations: smallest attainable p=1/21; exploratory, not familywise-corrected"})
    pd.DataFrame(null_summary).to_csv(HERE / "permutation_summary.csv", index=False)
    primary = result_frame[(result_frame.evaluation == "cross_dataset") &
                           (result_frame.feature_set == "all_compatible") &
                           (result_frame.model == "logistic")]
    great = bool(len(primary) >= 2 and (primary.balanced_accuracy >= .8).all() and
                 (primary.balanced_accuracy_CI_low > .5).all() and
                 all(r["balanced_accuracy_gain_full_minus_power"] > 0 for r in gain_rows))
    gain_supported = bool(len(datasets) >= 2 and all(r["gain_CI_low"] > 0 for r in gain_rows))
    verdict = {
        "predeclared_great_contrast_accuracy_met": great,
        "all_features_improve_over_power_with_positive_lower_CI_on_both_cohorts": gain_supported,
        "clinical_pain_prediction_validated": False,
        "absolute_cross_modal_physical_intensity_validated": False,
        "summary": ("Operational low/higher stimulus contrast threshold met; not clinical pain validation."
                    if great else "Great cross-dataset accuracy is NOT validated. Only one cohort had usable dose-labeled poststimulus EEG; the independent second-cohort stage is blocked."),
        "cohorts_with_measured_scores": len(datasets), "sampling": "10 presampled participants per cohort; usable counts in feature_summary.json",
        "interpretation": "Heat = measured Celsius, shock = presence of a fixed dose. Cross-cohort transfer scores apply only to the source-defined contrast. No mA or pain ratings were inferred.",
        "uncertainty": "Participant-cluster intervals conditional on each fitted training model; only two cohorts, retrospective holdouts, source-preprocessing and modality differences.",
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__, "fooof": "1.1.1"}
    }
    (HERE / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
