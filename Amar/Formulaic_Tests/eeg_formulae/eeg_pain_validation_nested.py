#!/usr/bin/env python3
"""
EEG-only pain prediction validation.
Input: a real, derived feature CSV, one row per participant-trial or participant-session.
Required columns: participant_id, pain_nrs, and numeric EEG feature columns.
Optional columns: dataset_id, trial_id, stimulus_intensity, pain_class.
This script does not download raw data or fabricate observations.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score,
    accuracy_score, classification_report)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

META = {"participant_id", "pain_nrs", "trial_id", "dataset_id", "pain_class", "stimulus_intensity"}

def metric_reg(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    rho = spearmanr(y, pred).statistic if len(np.unique(y)) > 1 and len(np.unique(pred)) > 1 else np.nan
    return {"n_rows": int(len(y)), "MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(mean_squared_error(y, pred)**0.5), "R2": float(r2_score(y, pred)),
            "Spearman_rho": float(rho) if np.isfinite(rho) else None}

def cluster_bootstrap_ci(y, pred, groups, metric=mean_absolute_error, n_boot=2000, seed=42):
    """Participant-cluster bootstrap: resample people, preserving all their rows."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": np.asarray(y), "pred": np.asarray(pred), "g": np.asarray(groups)})
    people = df.g.unique()
    vals = []
    if len(people) < 3:
        return [None, None]
    for _ in range(n_boot):
        sampled = rng.choice(people, size=len(people), replace=True)
        chunks = []
        for j, p in enumerate(sampled):
            c = df[df.g == p].copy()
            c["boot_cluster"] = j
            chunks.append(c)
        b = pd.concat(chunks, ignore_index=True)
        try:
            vals.append(metric(b.y, b.pred))
        except Exception:
            pass
    if not vals:
        return [None, None]
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]

def make_pipeline(model):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", RobustScaler()), ("model", model)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV of genuine derived EEG features and ratings")
    ap.add_argument("--out", default="eeg_pain_validation_out")
    ap.add_argument("--target", default="pain_nrs")
    ap.add_argument("--group", default="participant_id")
    ap.add_argument("--dataset", default="dataset_id")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--class-cut", type=float, default=None,
                    help="Optional pre-registered binary cutoff; scores >= cutoff are high")
    args = ap.parse_args()
    inp, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(inp)
    required = {args.group, args.target}
    if not required.issubset(df.columns):
        raise SystemExit(f"Required columns missing: {sorted(required - set(df.columns))}")
    eeg_cols = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    if not eeg_cols:
        raise SystemExit("No numeric EEG feature columns found. No data were generated.")
    df = df.dropna(subset=[args.group, args.target]).copy()
    if df[args.group].nunique() < max(3, args.n_splits):
        raise SystemExit("Too few unique participants for requested grouped CV; reduce n-splits only with justification.")
    X, y, groups = df[eeg_cols], df[args.target].astype(float), df[args.group]
    outer = GroupKFold(n_splits=min(args.n_splits, groups.nunique()))
    models = {
        "mean_baseline": make_pipeline(DummyRegressor(strategy="mean")),
        "ridge_all_EEG": make_pipeline(Ridge()),
        "elasticnet_all_EEG": make_pipeline(ElasticNet(max_iter=20000, random_state=42)),
        "random_forest_EEG": make_pipeline(RandomForestRegressor(
            n_estimators=400, min_samples_leaf=3, max_features=.7, random_state=42, n_jobs=-1)),
        "hist_gradient_boosting_EEG": make_pipeline(HistGradientBoostingRegressor(
            max_iter=150, l2_regularization=1.0, random_state=42))
    }
    grids = {
        "ridge_all_EEG": {"model__alpha": [.1, 1, 10, 100]},
        "elasticnet_all_EEG": {"model__alpha": [.001, .01, .1, 1], "model__l1_ratio": [.1, .5, .9]}
    }
    all_preds, summary = [], []
    for name, base in models.items():
        pred = np.full(len(df), np.nan)
        fold_info = []
        for fold, (tr, te) in enumerate(outer.split(X, y, groups), 1):
            estimator = clone(base)
            if name in grids:
                train_groups = groups.iloc[tr]
                inner_n = min(4, train_groups.nunique())
                if inner_n >= 2 and name != "mean_baseline":
                    inner = GroupKFold(n_splits=inner_n)
                    search = GridSearchCV(estimator, grids[name], cv=inner, scoring="neg_mean_absolute_error", n_jobs=-1)
                    search.fit(X.iloc[tr], y.iloc[tr], groups=train_groups)
                    estimator = search.best_estimator_
                else:
                    estimator.fit(X.iloc[tr], y.iloc[tr])
            else:
                estimator.fit(X.iloc[tr], y.iloc[tr])
            pred[te] = estimator.predict(X.iloc[te])
            fold_info.append({"fold": fold, "n_test_rows": int(len(te)),
                              "n_test_people": int(groups.iloc[te].nunique())})
        ok = np.isfinite(pred)
        met = metric_reg(y[ok], pred[ok])
        met.update({"model": name, "n_people": int(groups[ok].nunique()),
                    "MAE_cluster_bootstrap_95CI": cluster_bootstrap_ci(y[ok], pred[ok], groups[ok])})
        summary.append(met)
        all_preds.append(pd.DataFrame({"row_index": np.where(ok)[0], "participant_id": groups[ok].values,
                                       "observed": y[ok].values, "predicted": pred[ok], "model": name}))
    pred_df = pd.concat(all_preds, ignore_index=True)
    # Pre-specified label-permutation test for a fixed-alpha ridge model versus the mean baseline.
    # Permute ratings within participant to preserve each person's rating distribution. This assumes
    # exchangeability of trials within participant; for serially autocorrelated or condition-blocked
    # designs, use a block-/condition-constrained permutation instead.
    rng = np.random.default_rng(42)
    n_perm = max(0, int(args.permutations))
    folds = list(outer.split(X, y, groups))
    def oof_ridge_and_mean(yvec):
        rp = np.full(len(df), np.nan); mp = np.full(len(df), np.nan)
        for tr, te in folds:
            est = make_pipeline(Ridge(alpha=1.0))
            est.fit(X.iloc[tr], np.asarray(yvec)[tr])
            rp[te] = est.predict(X.iloc[te])
            mp[te] = float(np.mean(np.asarray(yvec)[tr]))
        return rp, mp
    ridge_oof, mean_oof = oof_ridge_and_mean(y.to_numpy())
    valid = np.isfinite(ridge_oof) & np.isfinite(mean_oof)
    observed_improvement = mean_absolute_error(y.to_numpy()[valid], mean_oof[valid]) - mean_absolute_error(y.to_numpy()[valid], ridge_oof[valid])
    null_improvements = []
    participant_indices = [np.flatnonzero(groups.to_numpy() == g) for g in groups.unique()]
    for _ in range(n_perm):
        yp = y.to_numpy().copy()
        for idx in participant_indices:
            yp[idx] = rng.permutation(yp[idx])
        rp, mp = oof_ridge_and_mean(yp)
        ok = np.isfinite(rp) & np.isfinite(mp)
        null_improvements.append(mean_absolute_error(yp[ok], mp[ok]) - mean_absolute_error(yp[ok], rp[ok]))
    p_perm = ((1 + sum(v >= observed_improvement for v in null_improvements)) / (n_perm + 1)) if n_perm else None
    (out / "ridge_permutation_test.json").write_text(json.dumps({
        "test": "within-participant label permutation; fixed Ridge alpha=1 versus fold-mean baseline",
        "n_permutations": n_perm,
        "observed_MAE_improvement_over_mean_baseline": float(observed_improvement),
        "one_sided_empirical_p": float(p_perm) if p_perm is not None else None,
        "null_improvements": [float(v) for v in null_improvements],
        "assumption": "trial labels exchangeable within participant; use constrained/block permutation if trial order or condition structure invalidates exchangeability",
        "note": "Run only on real data; this script does not generate observations."
    }, indent=2), encoding="utf-8")
    pred_df.to_csv(out / "grouped_cv_predictions.csv", index=False)
    pd.DataFrame(summary).to_json(out / "regression_metrics.json", orient="records", indent=2)
    pd.DataFrame(summary).to_csv(out / "regression_metrics.csv", index=False)
    (out / "run_metadata.json").write_text(json.dumps({
        "input": str(inp.resolve()), "n_rows": int(len(df)), "n_participants": int(groups.nunique()),
        "eeg_features": eeg_cols, "target": args.target, "group": args.group,
        "outer_cv": "GroupKFold", "nested_tuning": ["ridge_all_EEG", "elasticnet_all_EEG"],
        "warning": "Exploratory grouped CV, not external validation. If rows from multiple datasets exist, run leave-one-dataset-out as a separate locked test."
    }, indent=2), encoding="utf-8")
    # Visualization of out-of-fold predictions, never training-fit predictions.
    for name, sub in pred_df.groupby("model"):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(sub.observed, sub.predicted, alpha=.45, s=18)
        lo = min(sub.observed.min(), sub.predicted.min()); hi = max(sub.observed.max(), sub.predicted.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel("Observed pain NRS"); ax.set_ylabel("Out-of-fold predicted NRS")
        ax.set_title(f"{name}: grouped CV")
        fig.tight_layout(); fig.savefig(out / f"scatter_{name}.png", dpi=160); plt.close(fig)
    if args.class_cut is not None:
        yclass = (y >= args.class_cut).astype(int)
        # Participant-grouped CV with class model; binary cutoff must be selected before looking at results.
        class_models = {
            "majority_baseline": make_pipeline(DummyClassifier(strategy="most_frequent")),
            "logistic_EEG": make_pipeline(LogisticRegression(max_iter=5000, class_weight="balanced")),
            "random_forest_EEG": make_pipeline(RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                        class_weight="balanced", random_state=42, n_jobs=-1))
        }
        crows, csummary = [], []
        for name, model in class_models.items():
            pred = np.full(len(df), -1, dtype=int); prob = np.full(len(df), np.nan)
            for tr, te in outer.split(X, yclass, groups):
                est = clone(model); est.fit(X.iloc[tr], yclass.iloc[tr]); pred[te] = est.predict(X.iloc[te])
                if hasattr(est, "predict_proba") and len(est.named_steps["model"].classes_) == 2:
                    prob[te] = est.predict_proba(X.iloc[te])[:, 1]
            ok = pred >= 0
            d = {"model": name, "n_rows": int(ok.sum()),
                 "balanced_accuracy": float(balanced_accuracy_score(yclass[ok], pred[ok])),
                 "macro_F1": float(f1_score(yclass[ok], pred[ok], average="macro")),
                 "accuracy": float(accuracy_score(yclass[ok], pred[ok])),
                 "confusion_matrix_labels_0_1": confusion_matrix(yclass[ok], pred[ok], labels=[0,1]).tolist()}
            validprob = ok & np.isfinite(prob)
            d["AUROC"] = float(roc_auc_score(yclass[validprob], prob[validprob])) if len(np.unique(yclass[validprob])) == 2 else None
            csummary.append(d)
            crows.append(pd.DataFrame({"row_index": np.where(ok)[0], "participant_id": groups[ok].values,
                                       "observed_class": yclass[ok].values, "predicted_class": pred[ok],
                                       "prob_high": prob[ok], "model": name}))
        pd.DataFrame(csummary).to_json(out / "classification_metrics.json", orient="records", indent=2)
        pd.concat(crows, ignore_index=True).to_csv(out / "classification_predictions.csv", index=False)
    print(f"Finished analysis of supplied observed data: {len(df)} rows, {groups.nunique()} participants.")
    print(f"EEG features: {eeg_cols}")
    print(f"Outputs: {out.resolve()}")
    print("No raw EEG preprocessing is performed by this script; feature definitions/preprocessing must be documented separately.")

if __name__ == "__main__":
    main()
