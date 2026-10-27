#!/usr/bin/env python3
"""EEG-only pain-scale analysis pipeline. Fits only to supplied genuine trial-level EEG features.

Required input CSV columns: subject_id, trial_id, pain_nrs; optional session_id, trial_order,
condition, age_group, sex_group (for audit only), and numeric EEG feature columns.
No synthetic observations are generated. Participant-grouped nested CV is used for between-person
prediction. The separate within-person temporal analysis is descriptive and requires enough trials.
"""
from __future__ import annotations
import argparse, json, platform, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

ID_COLS = ["subject_id", "trial_id", "pain_nrs"]
DEFAULT_FEATURES = ["z_alpha", "z_beta", "z_theta", "z_gamma", "z_entropy", "z_faa", "z_connectivity", "z_erp", "z_n2", "z_p2", "z_peak_alpha"]

def clip_score(x): return np.clip(np.asarray(x, dtype=float), 0.0, 10.0)
def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -40, 40)
    return 1.0 / (1.0 + np.exp(-x))
def formula_alpha(z_alpha, intercept, slope): return 10.0 * sigmoid(intercept + slope*np.asarray(z_alpha, float))
def formula_linear(X, intercept, coefficients): return clip_score(intercept + np.asarray(X, float) @ np.asarray(coefficients, float))
def formula_temporal(raw_score, lam):
    if not 0 <= lam < 1: raise ValueError("lambda must be in [0,1)")
    x=clip_score(raw_score)
    if not len(x): return x
    out=np.empty_like(x); out[0]=x[0]
    for i in range(1,len(x)): out[i]=lam*out[i-1]+(1-lam)*x[i]
    return out

def robust_z(x, baseline):
    x=np.asarray(x,float); b=np.asarray(baseline,float); med=np.nanmedian(b); mad=np.nanmedian(np.abs(b-med))
    if not np.isfinite(mad) or mad <= np.finfo(float).eps: return np.full_like(x,np.nan)
    return np.clip((x-med)/(1.4826*mad),-5,5)

def score_metrics(y, p):
    y=np.asarray(y,float); p=clip_score(p); ok=np.isfinite(y)&np.isfinite(p); y=y[ok]; p=p[ok]
    if len(y)==0: return {"n":0}
    err=p-y; denom=np.sum((y-y.mean())**2)
    return {"n":int(len(y)),"MAE":float(np.mean(np.abs(err))),"RMSE":float(np.sqrt(np.mean(err**2))),
            "bias_prediction_minus_rating":float(np.mean(err)),"R2_out_of_sample":float(1-np.sum(err**2)/denom) if denom>0 else None,
            "Pearson_r":float(pearsonr(y,p).statistic) if len(y)>2 and np.std(y)>0 and np.std(p)>0 else None,
            "Spearman_rho":float(spearmanr(y,p).statistic) if len(y)>2 and np.std(y)>0 and np.std(p)>0 else None}

def validate_input(df, feature_cols):
    missing=set(ID_COLS)-set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not feature_cols: raise ValueError("No numeric EEG feature columns selected")
    if df["subject_id"].isna().any(): raise ValueError("subject_id contains missing values")
    df=df.copy(); df["pain_nrs"]=pd.to_numeric(df["pain_nrs"],errors="coerce")
    df=df.dropna(subset=["pain_nrs"])
    if not df.pain_nrs.between(0,10).all(): raise ValueError("pain_nrs must be observed values in [0,10]")
    if df["subject_id"].nunique()<5: warnings.warn("Fewer than five participants: grouped performance estimates will be unstable")
    for c in feature_cols: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df

def _model_bank():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.dummy import DummyRegressor
    # Fixed, prespecified settings keep runtime tractable; no data-dependent hyperparameter search is hidden.
    return {
      "mean_baseline": Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("m",DummyRegressor(strategy="mean"))]),
      "ridge_a10": Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler()),("m",Ridge(alpha=10.0))]),
      "elasticnet_a01_l1_05": Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler()),("m",ElasticNet(alpha=0.1,l1_ratio=0.5,max_iter=20000,random_state=42))]),
      "random_forest": Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("m",RandomForestRegressor(n_estimators=300,min_samples_leaf=5,max_features=0.8,n_jobs=-1,random_state=42))]),
      "hist_gradient_boosting": Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("m",HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,l2_regularization=1.0,random_state=42))])
    }

def grouped_cv(df, features, outdir, folds=5):
    from sklearn.model_selection import GroupKFold
    from sklearn.base import clone
    df=validate_input(df,features).reset_index(drop=True); df["row_id"]=np.arange(len(df)); groups=df.subject_id.astype(str); n_groups=groups.nunique()
    if n_groups<2: raise ValueError("At least two participants required")
    splitter=GroupKFold(n_splits=min(folds,n_groups)); X=df[features]; y=df.pain_nrs.to_numpy(float)
    predictions=[]; model_bank=_model_bank()
    for name, prototype in model_bank.items():
        pred=np.full(len(df),np.nan); fold_id=np.full(len(df),-1)
        for fold,(tr,te) in enumerate(splitter.split(X,y,groups)):
            model=clone(prototype); model.fit(X.iloc[tr],y[tr]); pred[te]=clip_score(model.predict(X.iloc[te])); fold_id[te]=fold
        tmp=df[["row_id","subject_id","trial_id","pain_nrs"]].copy(); tmp["fold"]=fold_id; tmp["model"]=name; tmp["prediction"]=pred
        predictions.append(tmp)
    pred_df=pd.concat(predictions,ignore_index=True); outdir.mkdir(parents=True,exist_ok=True)
    pred_df.to_csv(outdir/"grouped_oof_predictions.csv",index=False)
    summary=[]
    for name,g in pred_df.groupby("model",sort=False): summary.append({"model":name,**score_metrics(g.pain_nrs,g.prediction)})
    metrics_df=pd.DataFrame(summary); metrics_df.to_csv(outdir/"grouped_cv_metrics.csv",index=False)
    # Paired cluster bootstrap: sample participants, preserving all their trials; CI for MAE difference vs baseline.
    wide=pred_df.pivot(index=["row_id","subject_id","trial_id","pain_nrs"],columns="model",values="prediction").reset_index()
    subj=wide.subject_id.unique(); rng=np.random.default_rng(20261016); boot=[]
    base_mae=np.abs(wide.mean_baseline-wide.pain_nrs)
    for model in [m for m in wide.columns if m not in ["row_id","subject_id","trial_id","pain_nrs","mean_baseline"]]:
        diffs=[]
        for _ in range(2000):
            chosen=rng.choice(subj,size=len(subj),replace=True); chunks=[wide[wide.subject_id==s] for s in chosen]
            b=pd.concat(chunks,ignore_index=True)
            diffs.append(float(np.mean(np.abs(b[model]-b.pain_nrs)-np.abs(b.mean_baseline-b.pain_nrs))))
        point=float(np.mean(np.abs(wide[model]-wide.pain_nrs)-base_mae))
        boot.append({"model":model,"MAE_difference_vs_mean_baseline":point,"cluster_bootstrap_95CI_low":float(np.quantile(diffs,.025)),"cluster_bootstrap_95CI_high":float(np.quantile(diffs,.975)),"bootstrap_reps":2000})
    pd.DataFrame(boot).to_csv(outdir/"cluster_bootstrap_mae_difference.csv",index=False)
    # Subject-level descriptive metrics and subgroup audit where columns exist.
    sm=[]
    for (model,sid),g in pred_df.groupby(["model","subject_id"]): sm.append({"model":model,"subject_id":sid,**score_metrics(g.pain_nrs,g.prediction)})
    pd.DataFrame(sm).to_csv(outdir/"participant_level_metrics.csv",index=False)
    audit=[]
    for c in ["age_group","sex_group","condition","session_id"]:
        if c in df.columns:
            join_cols=["subject_id","trial_id"]
            meta=df[join_cols+[c]].drop_duplicates(join_cols)
            joined=pred_df.merge(meta,on=join_cols,how="left")
            for (model,grp),g in joined.groupby(["model",c],dropna=False): audit.append({"audit_variable":c,"group":str(grp),"model":model,**score_metrics(g.pain_nrs,g.prediction)})
    if audit: pd.DataFrame(audit).to_csv(outdir/"subgroup_audit.csv",index=False)
    # Feature association tests: Spearman pooled is explicitly descriptive; participant-cluster structure means not inferential.
    assoc=[]
    for f in features:
        xx=pd.to_numeric(df[f],errors="coerce"); ok=xx.notna()&df.pain_nrs.notna()
        if ok.sum()>2 and xx[ok].nunique()>1:
            r=spearmanr(xx[ok],df.loc[ok,"pain_nrs"]); assoc.append({"feature":f,"n_trials":int(ok.sum()),"pooled_spearman_rho_descriptive_only":float(r.statistic),"naive_p_value_not_valid_for_clustered_trials":float(r.pvalue)})
    pd.DataFrame(assoc).to_csv(outdir/"feature_associations_descriptive.csv",index=False)
    report={"status":"exploratory; not clinical validation","input_rows":int(len(df)),"participants":int(n_groups),"feature_columns":features,"outer_cv":"GroupKFold, participant held out; preprocessing fitted within each fold","metrics":summary,"important_caveat":"Pooled trial-level correlation p-values are not valid for repeated trials within participant; use participant-level or mixed-effects inference for confirmatory claims.","permutation_test":"Not run by default because a valid design-specific permutation scheme must be specified; use within-participant label permutation for trial-level association or participant-level permutation for between-person tests.","temporal_test":"Not run by default; must preserve chronological order and be evaluated separately from participant-held-out CV."}
    (outdir/"analysis_summary.json").write_text(json.dumps(report,indent=2))
    return report

def self_tests():
    assert np.allclose(clip_score([-2,0,5,10,13]),[0,0,5,10,10])
    assert np.allclose(formula_temporal([0,10,0],.5),[0,5,2.5])
    assert np.isnan(robust_z([1,2],[3,3])).all()
    assert np.all((formula_alpha(np.array([-100.,0.,100.]),0,1)>=0)&(formula_alpha(np.array([-100.,0.,100.]),0,1)<=10))
    try: formula_temporal([1],1.0); raise AssertionError("invalid lambda accepted")
    except ValueError: pass
    return {"software_unit_tests":"PASS","tests":["0-10 score bounds","EWMA recurrence","robust z zero-MAD handling","sigmoid score bounds","invalid smoothing parameter rejected"],"participant_data_tests":"NOT RUN until genuine feature CSV is supplied"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--features"); ap.add_argument("--outdir",default="eeg_pain_results"); ap.add_argument("--features-list",nargs="*",default=DEFAULT_FEATURES); ap.add_argument("--folds",type=int,default=5); ap.add_argument("--self-test",action="store_true"); args=ap.parse_args()
    result={"python":platform.python_version(),**self_tests()}
    if args.features:
        df=pd.read_csv(args.features); features=[c for c in args.features_list if c in df.columns]
        if not features: raise SystemExit("None of the requested feature names found; use --features-list with actual EEG feature columns")
        result["analysis"]=grouped_cv(df,features,Path(args.outdir),args.folds)
    else: result["data_analysis"] = "No EEG feature file supplied; no fitted coefficients or empirical performance are claimed."
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
