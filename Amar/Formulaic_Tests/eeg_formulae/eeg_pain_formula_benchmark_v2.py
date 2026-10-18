#!/usr/bin/env python3
"""
EEG-only interpretable pain formula benchmark, version 2.

IMPORTANT:
- Uses only observed rows supplied in an input feature table; creates no synthetic observations.
- Does not download data or claim a result until you run it on actual extracted EEG features.
- It tests formula families, not random forests/neural networks.
- Required CSV columns: participant_id, pain_nrs.
- Optional columns are used only when present. Feature definitions must be documented.
- Trial-wise EEG features must be extracted from real EEG with a fixed preprocessing pipeline.
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_1samp, wilcoxon
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

META = {"participant_id","pain_nrs","trial_id","dataset_id","site_id","session_id",
        "stimulus_intensity","stimulus_energy","pain_class","trial_order","condition"}

# Explicit interpretable families. The listed columns must already be calculated from genuine EEG.
SINGLE_FEATURES = {
    "evoked_rms": ["evoked_rms"],
    "n2p2_amp": ["n2p2_amp"],
    "alpha_suppression": ["alpha_suppression"],
    "theta_power": ["theta_power"],
    "alpha_power": ["alpha_power"],
    "beta_power": ["beta_power"],
    "gamma_power": ["gamma_power"],
    "peak_alpha_frequency": ["paf"],
    "spectral_entropy": ["spectral_entropy"],
    "aperiodic_exponent": ["aperiodic_exponent"],
    "theta_connectivity": ["theta_connectivity"],
    "alpha_gamma_pac": ["alpha_gamma_pac"],
    "microstate_C_duration": ["microstate_C_duration"],
    "microstate_D_duration": ["microstate_D_duration"],
    "erp_trial_variability": ["erp_trial_variability"],
    "habituation_slope": ["habituation_slope"],
}
FAMILIES = {
    "F_RMS_normalized": ["nrms_evoked"],
    "F_ERP": ["n2p2_amp", "n2_latency", "p2_latency"],
    "F_event_related_spectra": ["delta_log_theta", "delta_log_alpha", "delta_log_beta", "delta_log_gamma"],
    "F_multiband_spectral": ["theta_power", "alpha_power", "beta_power", "gamma_power", "paf"],
    "F_complexity_aperiodic": ["spectral_entropy", "permutation_entropy", "aperiodic_exponent", "aperiodic_offset"],
    "F_connectivity": ["theta_connectivity", "alpha_connectivity", "beta_connectivity"],
    "F_PAC": ["alpha_gamma_pac"],
    "F_microstates": ["microstate_C_duration", "microstate_D_duration", "microstate_transition_entropy"],
    "F_temporal_habituation": ["erp_trial_variability", "habituation_slope", "alpha_envelope_autocorr", "alpha_burst_rate"],
    "F_broad_EEG_only": [
        "nrms_evoked", "n2p2_amp", "delta_log_theta", "delta_log_alpha",
        "delta_log_beta", "delta_log_gamma", "theta_power", "alpha_power",
        "beta_power", "paf", "spectral_entropy", "aperiodic_exponent",
        "theta_connectivity", "alpha_gamma_pac", "microstate_C_duration",
        "microstate_D_duration", "erp_trial_variability", "habituation_slope"
    ],
}

def finite_frame(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

def fit_predict(train, test, target, cols, ridge_alpha=1.0):
    """OLS with training-only median imputation/scaling; ridge shrinkage optional."""
    from sklearn.linear_model import Ridge
    xtr, xte = finite_frame(train, cols), finite_frame(test, cols)
    med = xtr.median().fillna(0)
    xtr, xte = xtr.fillna(med), xte.fillna(med)
    mu = xtr.mean()
    sd = xtr.std(ddof=0).replace(0, 1).fillna(1)
    xtr, xte = (xtr-mu)/sd, (xte-mu)/sd
    y = pd.to_numeric(train[target], errors="coerce")
    valid = y.notna()
    if valid.sum() < max(5, len(cols)+2):
        raise ValueError("Too few training rows for this formula.")
    # Use OLS for <=4 predictors, ridge for larger equations to limit unstable coefficients.
    if len(cols) <= 4:
        model = sm.OLS(y[valid].to_numpy(), sm.add_constant(xtr.loc[valid], has_constant="add")).fit()
        pred = model.predict(sm.add_constant(xte, has_constant="add"))
        coeff = {"intercept": float(model.params.iloc[0])}
        coeff.update({c: float(model.params.iloc[i+1]) for i,c in enumerate(cols)})
        return np.asarray(pred), coeff
    model = Ridge(alpha=ridge_alpha)
    model.fit(xtr.loc[valid], y[valid])
    pred = model.predict(xte)
    coeff = {"intercept": float(model.intercept_)}
    coeff.update({c: float(v) for c,v in zip(cols, model.coef_)})
    return np.asarray(pred), coeff

def metrics(y, p):
    y, p = np.asarray(y,float), np.asarray(p,float)
    ok = np.isfinite(y)&np.isfinite(p); y,p=y[ok],p[ok]
    if len(y)==0: return {}
    r = pearsonr(y,p).statistic if len(y)>2 and np.std(y)>0 and np.std(p)>0 else np.nan
    rho = spearmanr(y,p).statistic if len(y)>2 and np.std(y)>0 and np.std(p)>0 else np.nan
    # Calibration slope is from observed ~ intercept + slope * prediction.
    slope = intercept = np.nan
    if len(y)>2 and np.std(p)>0:
        z=sm.OLS(y, sm.add_constant(p)).fit()
        intercept,slope=float(z.params[0]),float(z.params[1])
    return {"n_trials":int(len(y)), "MAE":float(mean_absolute_error(y,p)),
            "RMSE":float(mean_squared_error(y,p)**.5),
            "R2_test":float(r2_score(y,p)) if len(y)>1 else np.nan,
            "Pearson_r":float(r), "Spearman_rho":float(rho),
            "calibration_intercept":intercept, "calibration_slope":slope,
            "prediction_min":float(np.min(p)), "prediction_max":float(np.max(p))}

def cluster_bootstrap_mae(frame, group, y, p, B=2000, seed=2026):
    people = frame[group].dropna().unique()
    if len(people)<3: return [None,None]
    rng=np.random.default_rng(seed); vals=[]
    by={pid:frame.index[frame[group].eq(pid)].to_numpy() for pid in people}
    for _ in range(B):
        ids=rng.choice(people,len(people),replace=True)
        ix=np.concatenate([by[i] for i in ids])
        vals.append(mean_absolute_error(frame.loc[ix,y],frame.loc[ix,p]))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def within_person_tests(df, features, group, target):
    rows=[]
    for f in features:
        zs=[]
        for _,g in df.groupby(group):
            z=g[[target,f]].apply(pd.to_numeric,errors="coerce").dropna()
            if len(z)>=5 and z[target].nunique()>1 and z[f].nunique()>1:
                r=pearsonr(z[f],z[target]).statistic
                zs.append(np.arctanh(np.clip(r,-.999999,.999999)))
        if len(zs)>=3:
            t=ttest_1samp(zs,0)
            rows.append({"feature":f,"n_people":len(zs),
                         "mean_person_r":float(np.tanh(np.mean(zs))),
                         "mean_fisher_z":float(np.mean(zs)),
                         "t":float(t.statistic),"p_unadjusted":float(t.pvalue)})
    result=pd.DataFrame(rows)
    if len(result):
        # Benjamini-Hochberg FDR across the declared within-person feature screen.
        p=result["p_unadjusted"].to_numpy()
        order=np.argsort(p); m=len(p); q=np.empty(m); running=1.
        for rank_idx in range(m-1,-1,-1):
            rank=rank_idx+1
            running=min(running,p[order[rank_idx]]*m/rank)
            q[order[rank_idx]]=running
        result["q_BH"]=np.minimum(q,1)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,help="CSV with observed EEG features and pain ratings")
    ap.add_argument("--out",default="eeg_formula_v2_results")
    ap.add_argument("--group",default="participant_id")
    ap.add_argument("--target",default="pain_nrs")
    ap.add_argument("--ridge-alpha",type=float,default=1.0)
    ap.add_argument("--bootstrap",type=int,default=2000)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input)
    for c in [args.group,args.target]:
        if c not in df: raise SystemExit(f"Required column missing: {c}")
    df=df.dropna(subset=[args.group,args.target]).copy().reset_index(drop=True)
    df[args.target]=pd.to_numeric(df[args.target],errors="coerce")
    df=df.dropna(subset=[args.target]).reset_index(drop=True)
    npeople=df[args.group].nunique()
    if npeople<5: raise SystemExit("At least 5 independent participants are required for grouped evaluation.")
    # Derive ratios only from same-trial baseline/post-stimulus values; never compute a participant's
    # normalizer from all held-out pain outcomes or use target values in feature construction.
    if {"evoked_rms","spontaneous_rms"}.issubset(df.columns):
        # If a dedicated spontaneous baseline SD is supplied, prefer it; otherwise flag no normalized feature.
        if "spontaneous_rms_sd" in df:
            den=pd.to_numeric(df["spontaneous_rms_sd"],errors="coerce").replace(0,np.nan)
            df["nrms_evoked"]=(pd.to_numeric(df["evoked_rms"],errors="coerce")-
                               pd.to_numeric(df["spontaneous_rms"],errors="coerce"))/den
    if {"alpha_base","alpha_post"}.issubset(df.columns):
        vals=pd.concat([pd.to_numeric(df.alpha_base,errors="coerce"),pd.to_numeric(df.alpha_post,errors="coerce")])
        positive=vals[vals>0]
        eps=max(float(positive.median())*1e-12,1e-12) if len(positive) else 1e-12
        df["alpha_suppression"]=-np.log((pd.to_numeric(df.alpha_post,errors="coerce").clip(lower=0)+eps)/
                                         (pd.to_numeric(df.alpha_base,errors="coerce").clip(lower=0)+eps))
    if {"alpha_post","alpha_base"}.issubset(df.columns):
        df["delta_log_alpha"]=np.log(pd.to_numeric(df.alpha_post,errors="coerce").clip(lower=0)+1e-12)-np.log(pd.to_numeric(df.alpha_base,errors="coerce").clip(lower=0)+1e-12)
    if {"theta_post","theta_base"}.issubset(df.columns):
        df["delta_log_theta"]=np.log(pd.to_numeric(df.theta_post,errors="coerce").clip(lower=0)+1e-12)-np.log(pd.to_numeric(df.theta_base,errors="coerce").clip(lower=0)+1e-12)
    if {"beta_post","beta_base"}.issubset(df.columns):
        df["delta_log_beta"]=np.log(pd.to_numeric(df.beta_post,errors="coerce").clip(lower=0)+1e-12)-np.log(pd.to_numeric(df.beta_base,errors="coerce").clip(lower=0)+1e-12)
    if {"gamma_post","gamma_base"}.issubset(df.columns):
        df["delta_log_gamma"]=np.log(pd.to_numeric(df.gamma_post,errors="coerce").clip(lower=0)+1e-12)-np.log(pd.to_numeric(df.gamma_base,errors="coerce").clip(lower=0)+1e-12)
    if {"n2_amp","p2_amp"}.issubset(df.columns): df["n2p2_amp"]=pd.to_numeric(df.p2_amp,errors="coerce")-pd.to_numeric(df.n2_amp,errors="coerce")

    available={}
    for name,cols in {**SINGLE_FEATURES,**FAMILIES}.items():
        got=[c for c in cols if c in df.columns]
        if got: available[name]=got
    if not available: raise SystemExit("No recognized EEG feature columns found. See script docstring.")
    # Leave-one-participant-out. Every test participant is wholly held out.
    logo=LeaveOneGroupOut(); rows=[]; predrows=[]; coefrows=[]
    for tr,te in logo.split(df,df[args.target],groups=df[args.group]):
        train,test=df.iloc[tr],df.iloc[te]
        held=str(test[args.group].iloc[0])
        baseline=float(train[args.target].mean())
        for idx in te: predrows.append({"row_index":int(idx),args.group:held,"formula":"F0_training_mean","observed":float(df.iloc[idx][args.target]),"predicted":baseline})
        for name,cols in available.items():
            # Require enough complete-ish columns and limit predictor count to avoid underdetermined fits.
            if len(cols)>max(12, int(train[args.group].nunique()*.8)): continue
            try:
                p,coef=fit_predict(train,test,args.target,cols,args.ridge_alpha)
                for local,(idx,pp) in enumerate(zip(te,p)):
                    predrows.append({"row_index":int(idx),args.group:held,"formula":name,"observed":float(df.iloc[idx][args.target]),"predicted":float(pp)})
                for k,v in coef.items(): coefrows.append({"held_out_participant":held,"formula":name,"coefficient":k,"value":v})
            except Exception as e:
                warnings.warn(f"Skipping {name} for held-out {held}: {e}")
    pred=pd.DataFrame(predrows)
    pred.to_csv(out/"held_out_predictions.csv",index=False)
    for name,g in pred.groupby("formula"):
        m=metrics(g.observed,g.predicted); m["formula"]=name
        m["n_participants"]=int(g[args.group].nunique())
        m["MAE_cluster_bootstrap_95CI"]=cluster_bootstrap_mae(g,args.group,"observed","predicted",args.bootstrap)
        rows.append(m)
    met=pd.DataFrame(rows).sort_values("MAE")
    met.to_csv(out/"formula_metrics.csv",index=False)
    met.to_json(out/"formula_metrics.json",orient="records",indent=2)
    pd.DataFrame(coefrows).to_csv(out/"fold_coefficients.csv",index=False)
    # Person-specific scale audit: individual mean error, MAE, SD, observed range, prediction range,
    # within-person rank correlation and calibration slope. This is central to the user's question.
    scale=[]
    for (formula,pid),g in pred.groupby(["formula",args.group]):
        if formula=="F0_training_mean": continue
        m=metrics(g.observed,g.predicted)
        scale.append({"formula":formula,args.group:pid,"n_trials":m["n_trials"],
                      "observed_mean":float(g.observed.mean()),"predicted_mean":float(g.predicted.mean()),
                      "mean_bias_pred_minus_obs":float((g.predicted-g.observed).mean()),
                      "observed_sd":float(g.observed.std()),"predicted_sd":float(g.predicted.std()),
                      "observed_min":float(g.observed.min()),"observed_max":float(g.observed.max()),
                      "predicted_min":float(g.predicted.min()),"predicted_max":float(g.predicted.max()),
                      "MAE":m["MAE"],"RMSE":m["RMSE"],"within_person_r":m["Pearson_r"],
                      "within_person_rho":m["Spearman_rho"],"calibration_slope":m["calibration_slope"]})
    scales=pd.DataFrame(scale); scales.to_csv(out/"per_person_scale_audit.csv",index=False)
    singles=[f for f in SINGLE_FEATURES if f in df.columns]
    assoc=within_person_tests(df,singles,args.group,args.target)
    assoc.to_csv(out/"within_person_feature_tests.csv",index=False)
    # Plots are based solely on held-out predictions from supplied data.
    for formula,g in pred.groupby("formula"):
        fig,ax=plt.subplots(figsize=(6,5))
        ax.scatter(g.observed,g.predicted,s=12,alpha=.35)
        lo=min(g.observed.min(),g.predicted.min()); hi=max(g.observed.max(),g.predicted.max())
        ax.plot([lo,hi],[lo,hi],"k--",lw=1)
        ax.set(xlabel="Observed pain rating",ylabel="Held-out-participant predicted rating",title=formula)
        fig.tight_layout(); fig.savefig(out/f"{formula}_calibration.png",dpi=160); plt.close(fig)
    if len(scales):
        summary=scales.groupby("formula").agg(
            participants=(args.group,"nunique"),median_person_MAE=("MAE","median"),
            median_abs_person_bias=("mean_bias_pred_minus_obs",lambda x:float(np.median(np.abs(x)))),
            median_person_predicted_sd=("predicted_sd","median"),
            median_person_observed_sd=("observed_sd","median"),
            median_person_rho=("within_person_rho","median")).reset_index()
        summary.to_csv(out/"scale_profile_summary.csv",index=False)
    meta={"n_rows_observed":int(len(df)),"n_participants":int(npeople),
          "target":args.target,"group":args.group,"formulae_tested":list(available),
          "validation":"leave-one-participant-out; preprocessing/imputation/scaling fit on training participants",
          "data_integrity":"No synthetic rows or labels are created. This script does not fetch datasets.",
          "important_limits":[
              "The input must contain actual EEG-derived features and actual linked ratings.",
              "Per-person scores from held-out participants are evaluated without fitting to their ratings.",
              "A personalized 0-10 scale requires a calibration set from that person; it is not a new-person prediction.",
              "A formula using stimulus-evoked features must be compared with a stimulus-only baseline.",
              "Cross-dataset testing requires a dataset_id column and a separate leave-one-dataset-out run.",
              "The broad multifeature formula is exploratory; use nested grouped CV and pre-registration before confirmatory claims.",
              "Do not include demographics in the EEG score; use them for subgroup audits only."
          ]}
    (out/"run_metadata.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
    print("\nResults written. Metrics are empirical only if the input CSV contains real observations.")
if __name__=="__main__": main()
