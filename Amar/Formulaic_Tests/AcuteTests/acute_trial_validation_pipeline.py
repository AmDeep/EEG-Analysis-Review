#!/usr/bin/env python3
"""EEG-to-pain formula evaluation on real trial-level feature data.

Input data must be genuine, traceable EEG-derived features paired to individual
trial pain NRS ratings. This script does not synthesize rows. All scaling/imputation
and model fitting are performed within participant-grouped folds.

Expected CSV columns:
  subject_id, trial_id, pain_nrs, and >=1 numeric EEG predictor.
Recommended exact feature column names: p2_uv, n2_uv, z_alpha, z_beta, z_theta,
z_gamma, z_entropy, z_faa, z_connectivity, z_peak_alpha, z_n2, z_p2, z_erp.

Example:
  python acute_trial_validation_pipeline.py --input extracted_trial_features.csv --outdir trial_validation_results --folds 5 --bootstrap 2000 --feature-columns n2_uv p2_uv z_alpha z_beta

This script is an offline evaluation pipeline; it cannot extract features from raw
BDF/EEGLAB recordings without a dataset-specific event/epoch mapping. Do not treat
published aggregate means as trial rows to satisfy the input contract.
"""
from __future__ import annotations
import argparse, json, warnings, platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix,
    cohen_kappa_score, mean_absolute_percentage_error)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REQUIRED = ["subject_id", "trial_id", "pain_nrs"]
CATEGORY_LABELS = ["low_0_3", "moderate_4_6", "high_7_10"]

def clip10(x): return np.clip(np.asarray(x, dtype=float), 0, 10)

def categorize(y):
    # Explicit operational bins for evaluation only, not clinical cutoffs.
    y=np.asarray(y,float)
    return np.where(y < 4, 0, np.where(y < 7, 1, 2)).astype(int)

def metric_reg(y,p):
    y=np.asarray(y,float); p=clip10(p); e=p-y
    result={"n_trials":int(len(y)),"MAE":float(mean_absolute_error(y,p)),"RMSE":float(mean_squared_error(y,p)**0.5),
      "bias_prediction_minus_NRS":float(e.mean()),"R2_out_of_fold":float(r2_score(y,p)) if len(y)>1 and np.var(y)>0 else None,
      "Spearman_rho":float(spearmanr(y,p).statistic) if len(y)>2 and np.std(p)>0 and np.std(y)>0 else None,
      "Pearson_r":float(pearsonr(y,p).statistic) if len(y)>2 and np.std(p)>0 and np.std(y)>0 else None,
      "within_1_NRS_point":float(np.mean(np.abs(e)<=1)),"within_2_NRS_points":float(np.mean(np.abs(e)<=2))}
    yt=categorize(y); yp=categorize(p)
    result.update({"category_accuracy":float(accuracy_score(yt,yp)),"category_balanced_accuracy":float(balanced_accuracy_score(yt,yp)),
      "category_macro_F1":float(f1_score(yt,yp,labels=[0,1,2],average="macro",zero_division=0)),
      "ordinal_weighted_kappa":float(cohen_kappa_score(yt,yp,weights="quadratic")) if len(np.unique(yt))>1 else None})
    return result

def model_bank():
    return {
      "baseline_train_mean": Pipeline([("impute",SimpleImputer(strategy="median")),("model",DummyRegressor(strategy="mean"))]),
      "ridge_all_features_alpha10": Pipeline([("impute",SimpleImputer(strategy="median", add_indicator=True)),("scale",StandardScaler()),("model",Ridge(alpha=10.0))]),
      "elasticnet_all_features": Pipeline([("impute",SimpleImputer(strategy="median", add_indicator=True)),("scale",StandardScaler()),("model",ElasticNet(alpha=0.1,l1_ratio=0.5,max_iter=20000,random_state=42))]),
      "random_forest_all_features": Pipeline([("impute",SimpleImputer(strategy="median", add_indicator=True)),("model",RandomForestRegressor(n_estimators=300,min_samples_leaf=5,max_features=0.8,n_jobs=-1,random_state=42))]),
      "hist_gradient_boosting_all_features": Pipeline([("impute",SimpleImputer(strategy="median", add_indicator=True)),("model",HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,l2_regularization=1.0,random_state=42))]),
    }

def formula_feature_sets(columns):
    sets={}
    if "p2_uv" in columns: sets["F_P2_linear"]=["p2_uv"]
    if "n2p2_uv" in columns: sets["F_N2P2_linear"]=["n2p2_uv"]
    if "abs_n2_uv" in columns: sets["F_absN2_linear"]=["abs_n2_uv"]
    if "z_alpha" in columns:
        sets["F_alpha_linear_calibration"]=["z_alpha"]
        sets["F_alpha_sigmoid"]=["z_alpha"]
    for feat in ["z_beta","z_theta","z_gamma","z_entropy","z_faa","z_connectivity","z_peak_alpha","z_n2","z_p2","z_erp"]:
        if feat in columns: sets[f"F_{feat}"]=[feat]
    return sets

def run_grouped_cv(df, features, n_splits=5):
    groups=df.subject_id.astype(str).to_numpy(); y=df.pain_nrs.astype(float).to_numpy(); X=df[features]
    n_groups=len(np.unique(groups)); splits=min(n_splits,n_groups)
    if splits < 2: raise ValueError("Need at least two unique subjects for participant-held-out CV")
    splitter=GroupKFold(n_splits=splits)
    candidate_models=model_bank()
    for label, cols in formula_feature_sets(set(features)).items():
        candidate_models[label]=None if label == "F_alpha_sigmoid" else Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler()),("model",LinearRegression())])
    prediction_tables=[]; fold_assign=np.full(len(df),-1)
    for fold,(tr,te) in enumerate(splitter.split(X,y,groups)):
        fold_assign[te]=fold
        for name,proto in candidate_models.items():
            # For named single-feature formulae, fit on specified features only.
            feats=formula_feature_sets(set(features)).get(name,features)
            if name == "F_alpha_sigmoid":
                from scipy.optimize import least_squares
                xtr=pd.to_numeric(df.iloc[tr]["z_alpha"], errors="coerce").to_numpy(float)
                xte=pd.to_numeric(df.iloc[te]["z_alpha"], errors="coerce").to_numpy(float)
                med=float(np.nanmedian(xtr)); xtr=np.where(np.isfinite(xtr),xtr,med); xte=np.where(np.isfinite(xte),xte,med)
                mu=float(np.mean(xtr)); sd=float(np.std(xtr)); sd=sd if sd>1e-12 else 1.0
                xtr=(xtr-mu)/sd; xte=(xte-mu)/sd
                def sg(par, xx): return 10.0/(1.0+np.exp(-np.clip(par[0]+par[1]*xx,-40,40)))
                opt=least_squares(lambda par: sg(par,xtr)-y[tr], x0=np.array([0.0,1.0]), bounds=([-10.0,-10.0],[10.0,10.0]))
                pred=clip10(sg(opt.x,xte))
            else:
                mdl=clone(proto); mdl.fit(df.iloc[tr][feats],y[tr]); pred=clip10(mdl.predict(df.iloc[te][feats]))
            prediction_tables.append(pd.DataFrame({"row_index":te,"subject_id":groups[te],"trial_id":df.iloc[te].trial_id.astype(str).to_numpy(),
                "fold":fold,"model":name,"observed_pain_nrs":y[te],"predicted_pain_score":pred,"absolute_error":np.abs(pred-y[te]),
                "observed_category":np.array(CATEGORY_LABELS)[categorize(y[te])],"predicted_category":np.array(CATEGORY_LABELS)[categorize(pred)]}))
    pred_df=pd.concat(prediction_tables,ignore_index=True)
    metrics=[]
    for name,g in pred_df.groupby("model",sort=False): metrics.append({"model":name,**metric_reg(g.observed_pain_nrs,g.predicted_pain_score)})
    pred_df=pred_df.sort_values(["model","row_index"])
    return pred_df,pd.DataFrame(metrics)

def subject_cluster_bootstrap(pred, baseline_name="baseline_train_mean", reps=2000, seed=20261026):
    rng=np.random.default_rng(seed); wide=pred.pivot(index=["row_index","subject_id","trial_id","observed_pain_nrs"],columns="model",values="predicted_pain_score").reset_index()
    if baseline_name not in wide.columns: return pd.DataFrame()
    subjects=wide.subject_id.unique(); rows=[]
    other=[c for c in wide.columns if c not in {"row_index","subject_id","trial_id","observed_pain_nrs",baseline_name}]
    for model in other:
        point=float(np.mean(np.abs(wide[model]-wide.observed_pain_nrs)-np.abs(wide[baseline_name]-wide.observed_pain_nrs)))
        boot=[]
        for _ in range(reps):
            sampled=rng.choice(subjects,len(subjects),replace=True)
            parts=[wide[wide.subject_id==s] for s in sampled]
            b=pd.concat(parts,ignore_index=True)
            boot.append(float(np.mean(np.abs(b[model]-b.observed_pain_nrs)-np.abs(b[baseline_name]-b.observed_pain_nrs))))
        rows.append({"model":model,"MAE_difference_vs_baseline":point,"cluster_bootstrap_CI95_low":float(np.quantile(boot,.025)),
          "cluster_bootstrap_CI95_high":float(np.quantile(boot,.975)),"bootstrap_repetitions":reps,
          "interpretation":"negative difference favours candidate model"})
    return pd.DataFrame(rows)

def feature_associations(df, features):
    out=[]
    y=df.pain_nrs.astype(float)
    for f in features:
        x=pd.to_numeric(df[f],errors="coerce")
        ok=x.notna()&y.notna()
        if ok.sum()<4 or x[ok].nunique()<2: continue
        # Pooled association is descriptive because each person contributes repeated trials.
        rho=spearmanr(x[ok],y[ok])
        # Within-subject demeaned association is less confounded by stable between-person offsets.
        tmp=pd.DataFrame({"x":x[ok],"y":y[ok],"subject_id":df.loc[ok,"subject_id"].astype(str)})
        tmp["x_w"] = tmp.x - tmp.groupby("subject_id").x.transform("mean")
        tmp["y_w"] = tmp.y - tmp.groupby("subject_id").y.transform("mean")
        within=pearsonr(tmp.x_w,tmp.y_w) if tmp.x_w.std()>0 and tmp.y_w.std()>0 else None
        out.append({"feature":f,"n_trials":int(ok.sum()),"n_subjects":int(tmp.subject_id.nunique()),
          "pooled_spearman_rho_descriptive":float(rho.statistic),"pooled_naive_p_invalid_for_repeated_trials":float(rho.pvalue),
          "within_subject_centered_Pearson_r":float(within.statistic) if within else None,
          "within_subject_centered_naive_p_not_cluster_adjusted":float(within.pvalue) if within else None})
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--outdir",default="trial_validation_results"); ap.add_argument("--folds",type=int,default=5); ap.add_argument("--bootstrap",type=int,default=2000); ap.add_argument("--feature-columns",nargs="+",default=None,help="Explicit EEG-only feature columns. If omitted, infer columns from EEG feature naming patterns."); args=ap.parse_args()
    inp=Path(args.input); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(inp); missing=set(REQUIRED)-set(df.columns)
    if missing: raise SystemExit(f"Missing columns: {sorted(missing)}")
    df=df.dropna(subset=REQUIRED).copy(); df["pain_nrs"]=pd.to_numeric(df.pain_nrs,errors="coerce"); df=df.dropna(subset=["pain_nrs"])
    if "p2_uv" in df.columns and "n2_uv" in df.columns:
        df["n2p2_uv"] = pd.to_numeric(df["p2_uv"], errors="coerce") - pd.to_numeric(df["n2_uv"], errors="coerce")
        df["abs_n2_uv"] = pd.to_numeric(df["n2_uv"], errors="coerce").abs()
    if not df.pain_nrs.between(0,10).all(): raise SystemExit("pain_nrs must be between 0 and 10")
    if df.subject_id.nunique()<5: warnings.warn("Fewer than 5 subjects: all generalization estimates will be unstable")
    excluded=set(REQUIRED + ["trial_order","session_id","condition","stimulus_intensity","laser_power","age_group","sex_group","age","sex","timestamp","date","fold","run_id"])
    patterns=("n2","p2","alpha","beta","theta","gamma","entropy","connectivity","faa","erp","power","rms","mssd","peak","spectral","bandpower","z_")
    if args.feature_columns:
        absent=set(args.feature_columns)-set(df.columns)
        if absent: raise SystemExit(f"Requested feature columns missing: {sorted(absent)}")
        features=[]
        for c in args.feature_columns:
            df[c]=pd.to_numeric(df[c],errors="coerce")
            if df[c].notna().sum()>0 and df[c].nunique(dropna=True)>1: features.append(c)
    else:
        features=[]
        for c in df.columns:
            if c not in excluded and any(pat in c.lower() for pat in patterns):
                df[c]=pd.to_numeric(df[c],errors="coerce")
                if df[c].notna().sum()>0 and df[c].nunique(dropna=True)>1: features.append(c)
    for derived in ["n2p2_uv","abs_n2_uv"]:
        if derived in df.columns and derived not in features and df[derived].nunique(dropna=True)>1: features.append(derived)
    if not features: raise SystemExit("No variable numeric EEG feature columns found")
    pred,metrics=run_grouped_cv(df,features,args.folds)
    pred.to_csv(out/"grouped_oof_trial_predictions.csv",index=False)
    metrics.to_csv(out/"grouped_oof_formula_metrics.csv",index=False)
    subject_rows=[]
    for (model,subject),g in pred.groupby(["model","subject_id"]): subject_rows.append({"model":model,"subject_id":subject,**metric_reg(g.observed_pain_nrs,g.predicted_pain_score)})
    pd.DataFrame(subject_rows).to_csv(out/"subject_level_oof_metrics.csv",index=False)
    category=[]
    for name,g in pred.groupby("model",sort=False):
        yt=categorize(g.observed_pain_nrs); yp=categorize(g.predicted_pain_score); cm=confusion_matrix(yt,yp,labels=[0,1,2])
        for i,label in enumerate(CATEGORY_LABELS):
            n=int((yt==i).sum()); category.append({"model":name,"class":label,"n_true_trials":n,"recall":float(cm[i,i]/n) if n else None,
              "confusion_to_low":int(cm[i,0]),"confusion_to_moderate":int(cm[i,1]),"confusion_to_high":int(cm[i,2])})
    pd.DataFrame(category).to_csv(out/"pain_category_confusion_by_model.csv",index=False)
    subject_cluster_bootstrap(pred,reps=args.bootstrap).to_csv(out/"subject_cluster_bootstrap_mae_differences.csv",index=False)
    feature_associations(df,features).to_csv(out/"eeg_feature_rating_associations.csv",index=False)
    # Chronological within-subject future holdout (not between-subject generalization): last 20% of each subject's trials.
    if "trial_order" in df.columns:
        train_idx=[]; test_idx=[]
        for sid,g in df.groupby("subject_id"):
            g=g.sort_values("trial_order"); cut=max(1,int(np.floor(.8*len(g))))
            if len(g)-cut>=2: train_idx.extend(g.index[:cut]); test_idx.extend(g.index[cut:])
        if train_idx and test_idx:
            temporal=[]; X=df[features]; y=df.pain_nrs.astype(float)
            for name,proto in model_bank().items():
                mdl=clone(proto); mdl.fit(X.loc[train_idx],y.loc[train_idx]); p=clip10(mdl.predict(X.loc[test_idx]))
                temporal.append({"model":name,"test_trials":len(test_idx),"training_trials":len(train_idx),**metric_reg(y.loc[test_idx],p),"scope":"future trials from already-seen subjects"})
            pd.DataFrame(temporal).to_csv(out/"within_subject_temporal_holdout_metrics.csv",index=False)
    summary={"status":"completed on supplied genuine trial-level feature CSV","input_file":str(inp.resolve()),"n_trials":int(len(df)),"n_subjects":int(df.subject_id.nunique()),"features":features,
      "pain_category_cutoffs":"low 0-3; moderate 4-6; high 7-10 (operational analysis bins, not clinical cutoffs)","outer_validation":"GroupKFold; participant held out; preprocessing/model fitting restricted to training folds",
      "caveats":["Feature association pooled p-values are descriptive only; repeated trials are clustered by subject.","The aggregate formulas are not clinically validated by a single dataset.","If selecting a winner after comparing models, use a nested grouped validation or a genuinely independent external test set.","Between-subject prediction and within-subject temporal prediction are different tasks."],"unrun_tests":["permutation test is not implemented in this run script; use design-aware label shuffling that preserves subject and condition structure before confirmatory inference.","cross-dataset validation requires a second compatible trial-level feature dataset."]}
    (out/"trial_validation_run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"n_trials":len(df),"n_subjects":df.subject_id.nunique(),"features":features,"models":metrics.model.tolist(),"outputs":str(out.resolve())},indent=2))
if __name__=="__main__": main()
