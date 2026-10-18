#!/usr/bin/env python3
"""
EEG-only pain index research pipeline.

Input: real, observed trial/epoch-level feature table CSV; see report for schema.
This script does NOT fabricate observations and does NOT download datasets.
It fits multiple candidate models with participant-grouped CV and a held-out test set.
It is research code, not a medical device or clinical decision tool.
Dependencies: pandas, numpy, scikit-learn.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

FEATURE_BLOCKS = {
    "spectral": ["log_alpha_sm","log_beta_sm","log_theta_fz","log_gamma_sm",
                 "paf_hz","spectral_entropy","aperiodic_slope"],
    "evoked": ["n2p2_uv","n2_latency_ms","habituation_slope","trial_response_variability"],
    "network": ["connectivity_index","pac_index","alpha_envelope_cv"],
    "broad": ["log_alpha_sm","log_beta_sm","log_theta_fz","log_gamma_sm","paf_hz",
              "faa","alpha_envelope_cv","theta_cz_power","spectral_entropy",
              "aperiodic_slope","n2p2_uv","n2_latency_ms","habituation_slope",
              "connectivity_index","pac_index","trial_response_variability"]
}

def metrics(y, pred):
    return {"n": int(len(y)),
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** 0.5),
            "r2": float(r2_score(y, pred)) if len(y) > 1 else None,
            "spearman": float(pd.Series(y).corr(pd.Series(pred), method="spearman"))}

def build_models(seed=17):
    return {
        "F2_ridge_spectral": make_pipeline(SimpleImputer(strategy="median"), RobustScaler(),
                                          Ridge(alpha=10.0)),
        "F5_elastic_net_broad": make_pipeline(SimpleImputer(strategy="median"),
            StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20000)),
        "F8_random_forest_broad": make_pipeline(SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=400, min_samples_leaf=4,
                                  max_features=0.8, random_state=seed, n_jobs=-1)),
        "F8_hist_gradient_boost_broad": make_pipeline(SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05,
                                          l2_regularization=2.0, random_state=seed))
    }

def run(csv_path, output_dir, seed=17, test_size=0.2):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    required = {"participant_id", "pain_nrs"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must include {sorted(required)}")
    df = df[pd.to_numeric(df["pain_nrs"], errors="coerce").between(0,10)].copy()
    df["pain_nrs"] = df["pain_nrs"].astype(float)
    if df["participant_id"].nunique() < 5:
        raise ValueError("At least 5 participants required for a minimal grouped test; more is strongly recommended.")
    # Hold out participants, not trials.
    split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(split.split(df, groups=df["participant_id"]))
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    if set(train.participant_id) & set(test.participant_id):
        raise RuntimeError("Leakage: participant appears in train and test.")
    report = {"warning":"Exploratory results only. External validation requires a separate dataset.",
              "seed":seed, "rows":len(df), "participants":int(df.participant_id.nunique()),
              "train_rows":len(train), "test_rows":len(test),
              "train_participants":int(train.participant_id.nunique()),
              "test_participants":int(test.participant_id.nunique()), "models":{}}
    y_train = train.pain_nrs.to_numpy(); y_test = test.pain_nrs.to_numpy()
    report["models"]["mean_baseline"] = {"test":metrics(y_test, np.repeat(y_train.mean(),len(y_test)))}
    all_models = build_models(seed)
    for name, model in all_models.items():
        block = "spectral" if name == "F2_ridge_spectral" else "broad"
        cols = [c for c in FEATURE_BLOCKS[block] if c in df.columns]
        if len(cols) < 2:
            report["models"][name] = {"skipped":f"Only {len(cols)} compatible columns found"}
            continue
        model.fit(train[cols], y_train)
        pred = np.clip(model.predict(test[cols]), 0, 10)
        report["models"][name] = {"features":cols, "test":metrics(y_test,pred)}
        # Save predictions with actual labels, subject ID and model predictions.
        pd.DataFrame({"participant_id":test.participant_id.to_numpy(),
                      "pain_nrs_observed":y_test, "prediction":pred,
                      "residual":pred-y_test}).to_csv(output_dir/f"{name}_test_predictions.csv",index=False)
        # Record fitted coefficients where interpretable.
        final_est = model[-1]
        if hasattr(final_est,"coef_"):
            report["models"][name]["coefficients"] = {
                k:float(v) for k,v in zip(cols, np.asarray(final_est.coef_).ravel())}
            report["models"][name]["intercept"] = float(np.asarray(final_est.intercept_).ravel()[0])
    (output_dir/"metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True,help="Observed EEG feature table; no synthetic rows")
    p.add_argument("--out",default="eeg_pain_results")
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--test-size",type=float,default=.2)
    a=p.parse_args()
    result=run(a.csv,a.out,a.seed,a.test_size)
    print(json.dumps(result,indent=2))
if __name__=="__main__":
    main()
