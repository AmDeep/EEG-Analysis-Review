#!/usr/bin/env python3
"""
Interpretable EEG-only pain formula testing.
No random forest, neural network, boosting, or synthetic data.
Input CSV must contain observed participant/trial data:
participant_id, pain_nrs, and numeric EEG feature columns.
Optional feature columns: rms_evoked, rms_spontaneous, n2p2_amp, n2_amp,
p2_amp, alpha_base, alpha_post, theta_power, alpha_power, beta_power,
gamma_power, paf, theta_connectivity, spectral_entropy, aperiodic_exponent.
Optional metadata: trial_id, dataset_id, stimulus_intensity.
All feature names are examples; use only columns that actually exist.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_1samp, wilcoxon
import statsmodels.api as sm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

META={"participant_id","pain_nrs","trial_id","dataset_id","stimulus_intensity","pain_class"}

def metric(y,p):
    return {"n":int(len(y)),
            "MAE":float(mean_absolute_error(y,p)),
            "RMSE":float(mean_squared_error(y,p)**.5),
            "R2":float(r2_score(y,p)) if len(y)>1 else None,
            "Pearson_r":float(pearsonr(y,p).statistic) if len(y)>2 and np.std(p)>0 and np.std(y)>0 else None,
            "Spearman_rho":float(spearmanr(y,p).statistic) if len(y)>2 and np.std(p)>0 and np.std(y)>0 else None}

def get_feature_matrix(df, cols):
    X=df[cols].astype(float).copy()
    # Standardize only within training set, inside each held-out-participant fold.
    return X

def fit_ols_predict(train, test, target, cols):
    # Median imputation and standardization estimated only from training participants.
    med=train[cols].median()
    a=train[cols].fillna(med); b=test[cols].fillna(med)
    mu=a.mean(); sd=a.std(ddof=0).replace(0,1).fillna(1)
    a=(a-mu)/sd; b=(b-mu)/sd
    ma=sm.add_constant(a,has_constant="add")
    mb=sm.add_constant(b,has_constant="add")
    fit=sm.OLS(train[target].astype(float),ma).fit()
    return fit.predict(mb).to_numpy(), fit

def loo_grouped(df, target, cols, group="participant_id"):
    logo=LeaveOneGroupOut()
    preds=np.full(len(df),np.nan); records=[]
    for tr,te in logo.split(df[cols],df[target],groups=df[group]):
        train=df.iloc[tr]; test=df.iloc[te]
        # If too few distinct ratings or rows, OLS can still fit, but skip degenerate training sets.
        try:
            p,_=fit_ols_predict(train,test,target,cols)
            preds[te]=p
        except Exception as e:
            records.append({"held_out":str(test[group].iloc[0]),"error":str(e)})
    return preds,records

def cluster_boot_ci(df, ycol, pcol, group="participant_id", B=2000, seed=44):
    rng=np.random.default_rng(seed); people=df[group].dropna().unique()
    if len(people)<3: return [None,None]
    vals=[]
    for _ in range(B):
        ids=rng.choice(people,len(people),replace=True)
        chunks=[]
        for j, person in enumerate(ids):
            z=df[df[group]==person].copy()
            z["_boot_person"]=j
            chunks.append(z)
        b=pd.concat(chunks,ignore_index=True)
        vals.append(mean_absolute_error(b[ycol],b[pcol]))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--out",default="eeg_formula_results")
    ap.add_argument("--target",default="pain_nrs")
    ap.add_argument("--group",default="participant_id")
    ap.add_argument("--min-rating-for-piecewise",type=float,default=4.0,
                    help="Pre-specify the knot; used only if scientifically justified for dataset.")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input)
    for col in [args.group,args.target]:
        if col not in df: raise SystemExit(f"Missing required column: {col}")
    df=df.dropna(subset=[args.group,args.target]).copy().reset_index(drop=True)
    if df[args.group].nunique()<4: raise SystemExit("Need at least 4 distinct participants for grouped validation.")
    numeric=[c for c in df.select_dtypes(include=[np.number]).columns if c not in {args.target,"stimulus_intensity"}]
    # Formula definitions use only available features. No generated or imputed rows.
    candidates={}
    if "rms_evoked" in df: candidates["F1_raw_evoked_RMS"]=["rms_evoked"]
    if {"rms_evoked","rms_spontaneous"}.issubset(df.columns):
        # Feature calculated with training-independent participant baseline only if supplied baseline is legitimate.
        # If rms_spontaneous is trial-specific, normalization uses same row's pre-stimulus baseline.
        eps=np.finfo(float).eps
        df["nRMS_evoked"]=(df["rms_evoked"]-df["rms_spontaneous"])/df.groupby(args.group)["rms_spontaneous"].transform("std").replace(0,np.nan)
        candidates["F2_baseline_normalized_RMS"]=["nRMS_evoked"]
    if "n2p2_amp" in df: candidates["F3_N2P2_only"]=["n2p2_amp"]
    if {"n2_amp","p2_amp"}.issubset(df.columns):
        df["N2P2_difference"]=df["p2_amp"]-df["n2_amp"]
        candidates["F4_N2P2_difference"]=["N2P2_difference"]
    if {"alpha_base","alpha_post"}.issubset(df.columns):
        eps=max(float(np.nanmedian(df[["alpha_base","alpha_post"]].values[df[["alpha_base","alpha_post"]].values>0]))*1e-12 if (df[["alpha_base","alpha_post"]].values>0).any() else 1e-12,1e-12)
        df["alpha_suppression"]=-np.log((df["alpha_post"].clip(lower=0)+eps)/(df["alpha_base"].clip(lower=0)+eps))
        candidates["F5_alpha_suppression"]=["alpha_suppression"]
    bands=[c for c in ["theta_power","alpha_power","beta_power","gamma_power","paf","spectral_entropy","aperiodic_exponent","theta_connectivity"] if c in df]
    if bands: candidates["F6_multifeature_linear_OLS"]=bands
    if {"theta_power","alpha_power","beta_power"}.issubset(df.columns):
        candidates["F7_theta_alpha_beta_linear"]=["theta_power","alpha_power","beta_power"]
    results=[]; prediction_frames=[]
    # baseline predicts training-participant mean, scored on held-out person
    logo=LeaveOneGroupOut(); basepred=np.full(len(df),np.nan)
    for tr,te in logo.split(df,df[args.target],groups=df[args.group]):
        basepred[te]=df.iloc[tr][args.target].mean()
    bdf=df[[args.group,args.target]].copy(); bdf["prediction"]=basepred; bdf["formula"]="F0_training_mean_baseline"
    prediction_frames.append(bdf)
    results.append({"formula":"F0_training_mean_baseline",**metric(df[args.target],basepred)})
    for name,cols in candidates.items():
        if not cols: continue
        pred,errs=loo_grouped(df,args.target,cols,args.group)
        ok=np.isfinite(pred)
        if ok.sum()==0: continue
        tmp=df.loc[ok,[args.group,args.target]].copy(); tmp["prediction"]=pred[ok]; tmp["formula"]=name
        prediction_frames.append(tmp)
        m=metric(df.loc[ok,args.target],pred[ok]); m["formula"]=name; m["features"]=";".join(cols)
        m["n_participants"]=int(df.loc[ok,args.group].nunique())
        m["MAE_cluster_bootstrap_95CI"]=cluster_boot_ci(tmp,args.target,"prediction",args.group)
        # Residual-vs-rating correlation is diagnostic only, not a significance test for validity.
        results.append(m)
        (out/f"{name}_fold_errors.json").write_text(json.dumps(errs,indent=2))
    pred_df=pd.concat(prediction_frames,ignore_index=True)
    pred_df.to_csv(out/"out_of_participant_predictions.csv",index=False)
    pd.DataFrame(results).to_json(out/"formula_metrics.json",orient="records",indent=2)
    pd.DataFrame(results).to_csv(out/"formula_metrics.csv",index=False)
    # Participant-level association: estimate one correlation per participant where enough trials exist,
    # then test mean Fisher-z correlation across participants. This avoids trial-level pseudoreplication.
    association=[]
    for name,cols in candidates.items():
        if len(cols)!=1: continue
        vals=[]
        for pid,g in df.groupby(args.group):
            z=g[[args.target,cols[0]]].dropna()
            if len(z)>=5 and z[args.target].nunique()>1 and z[cols[0]].nunique()>1:
                r=pearsonr(z[cols[0]],z[args.target]).statistic
                r=np.clip(r,-.999999,.999999); vals.append(np.arctanh(r))
        if len(vals)>=3:
            t=ttest_1samp(vals,0)
            association.append({"formula":name,"n_participants_with_valid_r":len(vals),
                                "mean_fisher_z":float(np.mean(vals)),
                                "mean_r_backtransformed":float(np.tanh(np.mean(vals))),
                                "t_statistic":float(t.statistic),"p_uncorrected":float(t.pvalue),
                                "interpretation":"Within-person association; adjust for multiple tests if several features screened."})
    pd.DataFrame(association).to_csv(out/"within_person_association_tests.csv",index=False)
    # Visualize observed vs held-out predictions for each formula.
    for name,g in pred_df.groupby("formula"):
        fig,ax=plt.subplots(figsize=(5.4,5))
        ax.scatter(g[args.target],g["prediction"],alpha=.4,s=15)
        lo=min(g[args.target].min(),g["prediction"].min()); hi=max(g[args.target].max(),g["prediction"].max())
        ax.plot([lo,hi],[lo,hi],"--")
        ax.set_xlabel("Observed pain rating"); ax.set_ylabel("Held-out-participant predicted rating")
        ax.set_title(name); fig.tight_layout(); fig.savefig(out/f"{name}_observed_vs_predicted.png",dpi=160); plt.close(fig)
    meta={"n_rows":int(len(df)),"n_participants":int(df[args.group].nunique()),
          "target":args.target,"group":args.group,"available_numeric_features":numeric,
          "validation":"leave-one-participant-out; every participant held out once",
          "warning":"This is formula-based regression and descriptive statistical testing, not clinical validation. Include site/dataset external validation where possible. Baseline-normalization definitions must match actual recording design. Multiple-testing correction and design-specific permutation tests should be added before confirmatory claims."}
    (out/"metadata.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
    print("Finished. Only supplied rows were analyzed; no synthetic data were created.")

if __name__=="__main__": main()
