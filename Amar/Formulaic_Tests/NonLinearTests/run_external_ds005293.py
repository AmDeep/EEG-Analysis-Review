#!/usr/bin/env python3
"""Independent grouped validation on features extracted from OpenNeuro ds005293.

Input is the CSV exported by export_lep_trials.m. Observed NRS is the outcome;
only EEG-derived predictors enter the score formulas. Stimulus metadata is retained
only for a separate diagnostic baseline and never used by an EEG-only formula.
"""
from pathlib import Path
import argparse, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BANDS=['delta','theta','alpha','beta','gamma']

def score(y,p):
    return {'MAE':mean_absolute_error(y,p),'RMSE':mean_squared_error(y,p)**0.5,'R2':r2_score(y,p),
            'Pearson_r':pearsonr(y,p).statistic if np.std(p)>1e-12 else np.nan,
            'Spearman_rho':spearmanr(y,p).statistic if np.std(p)>1e-12 else np.nan,
            'prediction_SD':np.std(p,ddof=1),
            'calibration_slope_observed_on_predicted':np.polyfit(p,y,1)[0] if np.std(p)>1e-12 else np.nan,
            'calibration_intercept_observed_on_predicted':np.polyfit(p,y,1)[1] if np.std(p)>1e-12 else np.nan}

def build_sets(df):
    bresp=[f'log_{b}_resp' for b in BANDS]
    bbase=[f'log_{b}_base' for b in BANDS]
    bchg=[f'{b}_change_log' for b in BANDS]
    ratios=pd.DataFrame(index=df.index)
    for a,b in [('delta','alpha'),('theta','alpha'),('beta','alpha'),('gamma','beta'),('theta','beta')]:
        ratios[f'logratio_{a}_{b}']=df[f'log_{a}_resp']-df[f'log_{b}_resp']
    ratios['logratio_slow_fast']=df['log_slow_fast_response']
    p=np.column_stack([df[f'power_{b}_resp'].clip(lower=1e-12).to_numpy(float) for b in BANDS])
    rel=p/p.sum(1,keepdims=True); h=-(rel*np.log(np.maximum(rel,1e-12))).sum(1)/np.log(5)
    ratios['spectral_entropy_response']=h
    lep=['n2_amp','n2_latency_ms','p2_amp','p2_latency_ms','n2p2_peak2peak']
    sets={
      'response_log_band_power':df[bresp].copy(),
      'baseline_log_band_power':df[bbase].copy(),
      'baseline_relative_log_change':df[bchg].copy(),
      'single_change_delta':df[['delta_change_log']].copy(),
      'single_change_theta':df[['theta_change_log']].copy(),
      'single_change_alpha':df[['alpha_change_log']].copy(),
      'single_change_beta':df[['beta_change_log']].copy(),
      'single_change_gamma':df[['gamma_change_log']].copy(),
      'band_ratios_entropy':ratios.copy(),
      'laser_evoked_potential':df[lep].copy(),
      'bands_plus_LEP':df[bresp+bchg+lep].copy(),
      'broad_EEG_formula':df[bresp+bbase+bchg+lep].join(ratios)
    }
    for k,X in list(sets.items()): sets[k]=X.loc[:,~X.columns.duplicated()].replace([np.inf,-np.inf],np.nan)
    return sets

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='ds005293_trial_features.csv')
    ap.add_argument('--output-dir',default='external_ds005293_results')
    ap.add_argument('--outer-folds',type=int,default=5)
    args=ap.parse_args(); out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(args.input)
    required=['subject_id','trial_id','NRS']+[f'log_{b}_{w}' for b in BANDS for w in ['base','resp']]+[f'{b}_change_log' for b in BANDS]+['n2_amp','n2_latency_ms','p2_amp','p2_latency_ms','n2p2_peak2peak','log_slow_fast_response','spectral_entropy_response']+[f'power_{b}_resp' for b in BANDS]
    missing=sorted(set(required)-set(d.columns))
    if missing: raise ValueError(f'Missing required columns in {args.input}: {missing}. Generate with export_lep_trials.m and verify label extraction.')
    if d['subject_id'].isna().any() or d['NRS'].isna().any(): raise ValueError('Missing subject IDs or NRS labels; resolve label extraction first.')
    d['NRS']=pd.to_numeric(d['NRS'],errors='coerce'); d=d[d.NRS.between(0,10)].copy()
    if d.duplicated(['subject_id','trial_id']).any(): raise ValueError('Duplicate subject_id/trial_id keys. Resolve epoch duplication before model fitting.')
    y=d.NRS.to_numpy(float); groups=d.subject_id.astype(str).to_numpy()
    if pd.Series(groups).nunique()<args.outer_folds: raise ValueError('Not enough participants for requested grouped folds.')
    fx=build_sets(d); outer=GroupKFold(n_splits=args.outer_folds); base=np.full(len(d),np.nan)
    for tr,te in outer.split(np.zeros(len(d)),y,groups): base[te]=y[tr].mean()
    pred_rows=[]; metrics=[]; coef_rows=[]; prediction_cache={}
    metrics.append({'family':'training_mean_baseline','model':'MeanBaseline','n_features':0,**score(y,base)})
    prediction_cache[('training_mean_baseline','MeanBaseline')]=base
    for fam,X in fx.items():
        X=X.replace([np.inf,-np.inf],np.nan).copy()
        for mk in ['ridge','elasticnet']:
            p=np.full(len(d),np.nan)
            for fold,(tr,te) in enumerate(outer.split(X,y,groups),1):
                if mk=='ridge': est=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',Ridge())]); grid={'model__alpha':[0.1,1,10,100]}
                else: est=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',ElasticNet(max_iter=50000,tol=1e-4,random_state=20261023))]); grid={'model__alpha':[0.001,0.01,0.1],'model__l1_ratio':[0.3,0.7]}
                gs=GridSearchCV(est,grid,scoring='neg_mean_absolute_error',cv=GroupKFold(n_splits=3),n_jobs=1,refit=True)
                gs.fit(X.iloc[tr],y[tr],groups=groups[tr]); p[te]=np.clip(gs.predict(X.iloc[te]),0,10)
                coef_rows.extend([{'family':fam,'model':mk,'fold':fold,'feature':col,'coef_z':val} for col,val in zip(X.columns,gs.best_estimator_.named_steps['model'].coef_)])
            m=score(y,p);metrics.append({'family':fam,'model':mk,'n_features':X.shape[1],**m,'MAE_gain_vs_meanbaseline':score(y,base)['MAE']-m['MAE']})
            pred_rows.append(pd.DataFrame({'subject_id':groups,'trial_id':d.trial_id,'NRS':y,'prediction':p,'mean_baseline':base,'family':fam,'model':mk}))
            prediction_cache[(fam,mk)]=p
            print(f'{fam:28s} {mk:9s} MAE={m["MAE"]:.4f} RMSE={m["RMSE"]:.4f} R2={m["R2"]:.4f}',flush=True)
    mdf=pd.DataFrame(metrics).sort_values('MAE');mdf.to_csv(out/'external_metrics.csv',index=False)
    pd.concat(pred_rows,ignore_index=True).to_csv(out/'external_oof_predictions.csv',index=False)
    pd.DataFrame(coef_rows).to_csv(out/'external_fold_coefficients.csv',index=False)
    # Participant-cluster bootstrap (resample whole people; never treat trials as independent).
    boot=[]; rng=np.random.default_rng(20261023)
    per_subject=np.unique(groups)
    for fam,mk in prediction_cache:
        if fam=='training_mean_baseline': continue
        p=prediction_cache[(fam,mk)]
        per=[]
        for pid in per_subject:
            ix=(groups==pid)
            per.append((pid,mean_absolute_error(y[ix],p[ix])-mean_absolute_error(y[ix],base[ix])))
        diffs=np.array([v for _,v in per])
        reps=np.array([rng.choice(diffs,size=len(diffs),replace=True).mean() for _ in range(2000)])
        boot.append({'family':fam,'model':mk,'subjects':len(diffs),'mean_subject_MAE_difference_model_minus_baseline':float(diffs.mean()),
                     'CI95_low':float(np.quantile(reps,.025)),'CI95_high':float(np.quantile(reps,.975)),
                     'bootstrap_share_below_zero':float(np.mean(reps<0))})
    pd.DataFrame(boot).sort_values('mean_subject_MAE_difference_model_minus_baseline').to_csv(out/'external_cluster_bootstrap.csv',index=False)
    audit={'dataset':'OpenNeuro ds005293 v1.0.0 / NEMAR on005293 v1.0.0','rows':int(len(d)),'subjects':int(pd.Series(groups).nunique()),'NRS_range':[float(d.NRS.min()),float(d.NRS.max())],
           'outer_cv':f'{args.outer_folds}-fold GroupKFold by subject','inner_cv':'3-fold GroupKFold within outer training set','outcome':'observed single-trial NRS, 0..10',
           'score_predictors':'EEG band power, baseline-relative band power, spectral ratios/entropy and LEP N2/P2 amplitudes/latencies only',
           'stimulus_intensity':'not used as an EEG formula predictor; if available, test it in a separate stimulus-only comparator','warning':'Confirm that exported ratings reflect observed per-trial NRS, not nominal calibrated categories, and confirm epoch-to-rating alignment before interpreting performance.'}
    (out/'external_audit.json').write_text(json.dumps(audit,indent=2))
    print('\nTop formulas:');print(mdf.head(12).to_string(index=False));print('Saved results in',out)
if __name__=='__main__':
    warnings.filterwarnings('ignore'); main()
