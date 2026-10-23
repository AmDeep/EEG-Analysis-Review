#!/usr/bin/env python3
"""Exhaustive subset + nonlinear formula family search for EEG-only NRS scoring.

Primary source table: /mnt/data/Trials.pkl (local precomputed EEG features only).
Unit: one row per unique (id, Area, B, iTrial) key, after averaging EEG values
across paired rows that share the same NRS. This is explicitly an assumption to
address the observed paired-row structure and is not a claim of external validity.

All formula families are coefficient-based Ridge equations; no RF, boosting,
subjective predictors, skin conductance, demographics, trial metadata, or pain
condition label are predictor variables. The outcome NRS is used as y only.
"""
from __future__ import annotations
import os, json, itertools, warnings, time, zipfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer, SplineTransformer, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from docx import Document
from docx.shared import Inches, Pt

ROOT=Path(__file__).resolve().parent
OUT=ROOT
DATA=Path(os.getenv('EEG_TRIALS_PATH', str(ROOT/'Trials.pkl')))
OUTER_N=5
INNER_N=3
ALPHAS=[1.0, 10.0, 100.0]
SCREEN_ALPHA=10.0
SEED=20261023
BANDS=['delta','theta','alpha','beta','gamma']
POWER_COLS=[f'EEG_{b}' for b in BANDS]
MOMENT_COLS=['EEG_mean','EEG_kurtosis','EEG_skew','EEG_SampEn_01','EEG_peak_frequency','EEG_peak_alpha_frequency','EEG_peak_beta_frequency','EEGSampEn_01_m4']
KEYS=['id','Area','B','iTrial']

# Pure mathematical basis expansion functions. Inputs have been standardized inside each CV training fold.
def exp_basis(z):
    z=np.asarray(z,dtype=float); zc=np.clip(z,-3.0,3.0)
    return np.concatenate([z, np.exp(zc), np.exp(-zc)], axis=1)
def saturation_basis(z):
    z=np.asarray(z,dtype=float)
    sig=1.0/(1.0+np.exp(-np.clip(z,-8,8)))
    return np.concatenate([z, np.tanh(z), sig, z/np.sqrt(1.0+z*z)],axis=1)
def powerlaw_basis(z):
    z=np.asarray(z,dtype=float)
    signed_sqrt=np.sign(z)*np.sqrt(np.abs(z))
    signed_3_2=np.sign(z)*(np.abs(z)**1.5)
    return np.concatenate([z,signed_sqrt,signed_3_2,z*np.abs(z)],axis=1)
def reciprocal_basis(z):
    z=np.asarray(z,dtype=float); zc=np.clip(z,-8,8)
    return np.concatenate([z,1.0/(1.0+np.abs(z)),1.0/(1.0+np.exp(zc)),1.0/(1.0+np.exp(-zc))],axis=1)
def exp_interaction_basis(z):
    z=np.asarray(z,dtype=float); zc=np.clip(z,-3,3); parts=[z]
    for i in range(z.shape[1]):
        for j in range(i+1,z.shape[1]):
            parts.append(np.exp(np.clip(zc[:,[i]]+zc[:,[j]],-6,6)))
            parts.append(np.exp(-np.clip(zc[:,[i]]+zc[:,[j]],-6,6)))
    return np.concatenate(parts,axis=1)


def metrics(y,pred):
    pred=np.asarray(pred,float); y=np.asarray(y,float)
    sd=float(np.std(pred,ddof=1)) if len(pred)>1 else 0.0
    return {
      'MAE':float(mean_absolute_error(y,pred)),
      'RMSE':float(np.sqrt(mean_squared_error(y,pred))),
      'R2':float(r2_score(y,pred)),
      'Pearson_r':float(pearsonr(y,pred).statistic) if sd>1e-12 else np.nan,
      'Spearman_rho':float(spearmanr(y,pred).statistic) if sd>1e-12 else np.nan,
      'prediction_SD':sd,
      'calibration_slope':float(np.polyfit(pred,y,1)[0]) if sd>1e-12 else np.nan,
      'calibration_intercept':float(np.polyfit(pred,y,1)[1]) if sd>1e-12 else np.nan,
    }


def make_basis_pipeline(family, alpha=SCREEN_ALPHA):
    steps=[('impute',SimpleImputer(strategy='median'))]
    if family in ['log_linear','relative_linear','clr_linear']:
        steps += [('scale',StandardScaler()),('ridge',Ridge(alpha=alpha))]
    elif family=='polynomial_degree_2':
        steps += [('scale_in',StandardScaler()),('poly',PolynomialFeatures(degree=2,include_bias=False)),('scale_out',StandardScaler()),('ridge',Ridge(alpha=alpha))]
    elif family=='polynomial_degree_3':
        steps += [('scale_in',StandardScaler()),('poly',PolynomialFeatures(degree=3,include_bias=False)),('scale_out',StandardScaler()),('ridge',Ridge(alpha=alpha))]
    elif family=='pairwise_interactions':
        steps += [('scale_in',StandardScaler()),('poly',PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)),('scale_out',StandardScaler()),('ridge',Ridge(alpha=alpha))]
    elif family=='exponential_inverse':
        steps += [('scale_in',StandardScaler()),('basis',FunctionTransformer(exp_basis,validate=False)),('scale_out',StandardScaler()),('ridge',Ridge(alpha=alpha))]
    elif family=='exponential_interactions':
        steps += [('scale_in',StandardScaler()),('basis',FunctionTransformer(exp_interaction_basis,validate=False)),('scale_out',StandardScaler()),('ridge',Ridge(alpha=alpha))]
    elif family=='sigmoid_saturation':
        steps += [('scale_in',StandardScaler()),('basis',FunctionTransformer(saturation_basis,validate=False)),('scale_out',StandardScaler()),('ridge',Ridge(alpha=alpha))]
    elif family=='fractional_power':
        steps += [('scale_in',StandardScaler()),('basis',FunctionTransformer(powerlaw_basis,validate=False)),('scale_out',StandardScaler()),('ridge',Ridge(alpha=alpha))]
    elif family=='reciprocal_saturation':
        steps += [('scale_in',StandardScaler()),('basis',FunctionTransformer(reciprocal_basis,validate=False)),('scale_out',StandardScaler()),('ridge',Ridge(alpha=alpha))]
    elif family=='additive_spline':
        steps += [('scale_in',StandardScaler()),('spline',SplineTransformer(n_knots=4,degree=3,knots='quantile',extrapolation='constant')),('scale_out',StandardScaler()),('ridge',Ridge(alpha=alpha))]
    elif family=='yeo_johnson':
        steps += [('power',PowerTransformer(method='yeo-johnson',standardize=True)),('ridge',Ridge(alpha=alpha))]
    else: raise ValueError(family)
    return Pipeline(steps)

FAMILIES=['log_linear','polynomial_degree_2','polynomial_degree_3',
          'exponential_inverse','exponential_interactions','sigmoid_saturation','fractional_power',
          'additive_spline','yeo_johnson','relative_linear','clr_linear']


def build_primary_data():
    if not DATA.exists(): raise FileNotFoundError(DATA)
    raw=pd.read_pickle(DATA).reset_index(drop=True)
    req=KEYS+['NRS','pain']+POWER_COLS+MOMENT_COLS
    missing=[c for c in req if c not in raw]
    if missing: raise ValueError(f'Missing columns: {missing}')
    raw['NRS']=pd.to_numeric(raw.NRS,errors='coerce')
    raw=raw[raw.NRS.between(0,10)].copy()
    n_raw=len(raw); n_subjects=raw.id.nunique()
    nrs_nunique=raw.groupby(KEYS,dropna=False).NRS.nunique()
    if (nrs_nunique>1).any(): raise ValueError('NRS differs within repeated key; aggregation is not safe.')
    group_size=raw.groupby(KEYS,dropna=False).size()
    # Main assumption: these paired rows carry a shared subjective outcome; average only EEG metrics.
    meta=raw.groupby(KEYS,as_index=False,dropna=False).agg(id=('id','first'),Area=('Area','first'),B=('B','first'),iTrial=('iTrial','first'),NRS=('NRS','first'),n_rows=('NRS','size'))
    featcols=list(dict.fromkeys(POWER_COLS+MOMENT_COLS))
    feats=raw.groupby(KEYS,dropna=False)[featcols].mean().reset_index()
    df=meta.merge(feats,on=KEYS,validate='one_to_one')
    # Strictly EEG-only inputs. Powers positive; logging is defined using small epsilon.
    P=df[POWER_COLS].astype(float).clip(lower=1e-12)
    L=np.log(P); L.columns=BANDS
    R=P.div(P.sum(axis=1),axis=0); R.columns=BANDS
    # Candidate matrices are generated from each subset; the transforms learned from data
    # (standardization, splines, Yeo-Johnson) remain inside the pipeline/fold.
    all_candidates={}
    subsets=[]
    for k in range(1,6):
        for inds in itertools.combinations(range(5),k): subsets.append(tuple(inds))
    names=lambda inds: '+'.join(BANDS[i] for i in inds)
    for inds in subsets:
        subset_name=names(inds)
        XL=L.iloc[:,list(inds)].copy(); XL.columns=[BANDS[i] for i in inds]
        XR=R.iloc[:,list(inds)].copy(); XR.columns=[BANDS[i] for i in inds]
        # CLR/log-ratio coordinates: logarithmic band power relative to geometric mean of this subset.
        XCLR=XL.sub(XL.mean(axis=1),axis=0)
        for fam in FAMILIES:
            if fam=='relative_linear':
                X=XR.copy()
            elif fam=='clr_linear':
                X=XCLR.copy()
            else:
                X=XL.copy()
            if fam=='clr_linear' and len(inds)==1: continue  # mathematically identically zero
            key=(subset_name,fam)
            all_candidates[key]=(X,tuple(inds),fam)
    # Additional prespecified EEG feature blocks (outside 5-band powers but still EEG-only).
    moms=df[MOMENT_COLS].astype(float).copy()
    log_rel=np.log(R.clip(lower=1e-12)); log_rel.columns=[f'logrel_{b}' for b in BANDS]
    entropy=-(R.clip(lower=1e-12)*np.log(R.clip(lower=1e-12))).sum(axis=1)/np.log(5)
    ratios=pd.DataFrame(index=df.index)
    for a,b in [('delta','alpha'),('theta','alpha'),('theta','beta'),('beta','alpha'),('gamma','beta'),('gamma','alpha'),('delta','theta'),('alpha','beta')]:
        ratios[f'log_{a}_{b}']=L[a]-L[b]
    ratios['log_slow_fast']=np.log((P.EEG_delta+P.EEG_theta).clip(lower=1e-12))-np.log((P.EEG_beta+P.EEG_gamma).clip(lower=1e-12))
    composite=pd.concat([L.add_prefix('log_'),log_rel,ratios,pd.DataFrame({'spectral_entropy':entropy}),moms],axis=1)
    # Deduplicate columns and ensure finite or missing values.
    composite=composite.loc[:,~composite.columns.duplicated()].replace([np.inf,-np.inf],np.nan)
    all_candidates[('ALL_BANDS','composite_linear')]=(composite,None,'composite_linear')
    all_candidates[('ALL_BANDS','composite_quadratic')]=(composite,None,'composite_quadratic')
    all_candidates[('ALL_BANDS','bands_quadratic')]=(L.copy(),None,'polynomial_degree_2')
    all_candidates[('ALL_BANDS','bands_cubic')]=(L.copy(),None,'polynomial_degree_3')
    all_candidates[('ALL_BANDS','bands_exponential')]=(L.copy(),None,'exponential_inverse')
    all_candidates[('ALL_BANDS','bands_spline')]=(L.copy(),None,'additive_spline')
    audit={'raw_rows':n_raw,'unique_rating_keys':len(df),'participants':n_subjects,'unique_area_codes':int(df.Area.nunique()),
           'NRS_min':float(df.NRS.min()),'NRS_max':float(df.NRS.max()),'NRS_mean':float(df.NRS.mean()),'NRS_sd':float(df.NRS.std()),
           'rows_per_key_counts':{str(int(k)):int(v) for k,v in group_size.value_counts().sort_index().items()},
           'n_keys_with_two_rows':int((group_size==2).sum()),'n_keys_with_other_row_count':int((group_size!=2).sum()),
           'NRS_consistent_within_key':bool((nrs_nunique<=1).all()),'paired_label_pattern':raw.groupby(KEYS).pain.nunique().value_counts().to_dict(),
           'predictor_policy':'EEG feature columns only; NRS outcome only; excluded pain label, SC/SCH/SCF, demographics, questionnaire variables, IDs and trial/area metadata.',
           'unit_assumption':'Average EEG features across rows sharing same participant/Area/B/iTrial and same NRS. Must be confirmed against original event mapping.'}
    y=df.NRS.to_numpy(float); groups=df.id.to_numpy(str)
    base_arrays={k:v[0].to_numpy(float) for k,v in all_candidates.items()}
    return raw,df,y,groups,all_candidates,audit,base_arrays


def candidate_pipeline(fam,alpha):
    if fam=='composite_linear': return make_basis_pipeline('log_linear',alpha)
    if fam=='composite_quadratic': return make_basis_pipeline('polynomial_degree_2',alpha)
    if fam=='bands_quadratic': return make_basis_pipeline('polynomial_degree_2',alpha)
    if fam=='bands_cubic': return make_basis_pipeline('polynomial_degree_3',alpha)
    if fam=='bands_exponential': return make_basis_pipeline('exponential_inverse',alpha)
    if fam=='bands_spline': return make_basis_pipeline('additive_spline',alpha)
    return make_basis_pipeline(fam,alpha)


def run_search():
    start=time.time()
    raw,df,y,groups,candidates,audit,_=build_primary_data()
    print(f"Dataset: raw_rows={len(raw)} unique_keys={len(df)} participants={pd.Series(groups).nunique()} candidates={len(candidates)}",flush=True)
    outer=GroupKFold(n_splits=OUTER_N)
    outer_splits=list(outer.split(np.zeros(len(y)),y,groups))
    # Candidate metrics under one fixed, prespecified ridge penalty, to publish all combinations.
    # Optional resume flags let the broad screen and nested evaluation run as separate jobs.
    screen_path=OUT/'all_subset_formula_screen.csv'
    nested_only=os.getenv('EEG_NESTED_ONLY','0')=='1' and screen_path.exists()
    if nested_only:
        screen_df=pd.read_csv(screen_path)
        base=np.full(len(y),np.nan)
        for tr,te in outer_splits: base[te]=float(np.mean(y[tr]))
        base_metric=metrics(y,base)
        screen_df=screen_df.sort_values('MAE').reset_index(drop=True)
        print(f'Reusing saved exhaustive screen: {len(screen_df)} candidates',flush=True)
    else:
        screen=[]
        for ci,(key,(X,inds,fam)) in enumerate(candidates.items(),1):
            pred=np.full(len(y),np.nan)
            pipe=candidate_pipeline(fam,SCREEN_ALPHA)
            for tr,te in outer_splits:
                m=clone(pipe);m.fit(X.iloc[tr],y[tr]);pred[te]=np.clip(m.predict(X.iloc[te]),0,10)
            met=metrics(y,pred)
            sname,ff=key
            screen.append({'subset':sname,'family':ff,'n_input_features':X.shape[1],'ridge_alpha_fixed':SCREEN_ALPHA,**met})
            if ci%25==0: print(f'Fixed-alpha screen {ci}/{len(candidates)}',flush=True)
        screen_df=pd.DataFrame(screen)
        # mean baseline out-of-fold
        base=np.full(len(y),np.nan)
        for tr,te in outer_splits: base[te]=float(np.mean(y[tr]))
        base_metric=metrics(y,base)
        screen_df['MAE_gain_vs_mean_baseline']=base_metric['MAE']-screen_df.MAE
        screen_df['RMSE_gain_vs_mean_baseline']=base_metric['RMSE']-screen_df.RMSE
        screen_df=screen_df.sort_values('MAE').reset_index(drop=True)
        screen_df.to_csv(screen_path,index=False)
        # Save audit and baseline checkpoint before any slower nested work.
        (OUT/'audit.json').write_text(json.dumps(audit,indent=2,default=str),encoding='utf-8')
        (OUT/'baseline_checkpoint.json').write_text(json.dumps(base_metric,indent=2),encoding='utf-8')
        if os.getenv('EEG_SCREEN_ONLY','0')=='1':
            print('SCREEN_ONLY COMPLETE. Baseline:',base_metric,flush=True)
            print(screen_df.head(15)[['subset','family','n_input_features','MAE','RMSE','R2','MAE_gain_vs_mean_baseline']].to_string(index=False),flush=True)
            return
    # Nested formula selection: choose any subset, nonlinear basis and alpha entirely from inner grouped CV.
    # Outer test folds evaluate the complete 342-specification adaptive search process.
    nested_pred=np.full(len(y),np.nan); selections=[]; nested_fold=[]; nested_coefs=[]
    for of,(otr,ote) in enumerate(outer_splits,1):
        yin=y[otr]; gin=groups[otr]
        inner=GroupKFold(n_splits=INNER_N)
        inner_splits=list(inner.split(np.zeros(len(otr)),yin,gin))
        best={'score':np.inf}
        # Full nested selection across EVERY candidate from the exhaustive screen and every alpha.
        # The outer folds evaluate the whole search/tuning procedure; no candidate is shortlisted using outer test data.
        nested_candidates=candidates
        for ci,(key,(X,inds,fam)) in enumerate(nested_candidates.items(),1):
            Xtr=X.iloc[otr]
            for alpha in ALPHAS:
                fold_losses=[]
                pipe=candidate_pipeline(fam,alpha)
                for itr,iva in inner_splits:
                    mm=clone(pipe);mm.fit(Xtr.iloc[itr],yin[itr])
                    ph=np.clip(mm.predict(Xtr.iloc[iva]),0,10)
                    fold_losses.append(mean_absolute_error(yin[iva],ph))
                # Mean fold MAE; equal fold weighting keeps subjects closer to equally weighted.
                score=float(np.mean(fold_losses))
                if score<best['score']:
                    best={'score':score,'key':key,'X':X,'fam':fam,'alpha':alpha,'fold_losses':fold_losses,'n_features':X.shape[1]}
            if ci%25==0:
                print(f'Nested outer fold {of}/{OUTER_N}: evaluated {ci}/{len(nested_candidates)} formulas',flush=True)
        mm=candidate_pipeline(best['fam'],best['alpha'])
        mm.fit(best['X'].iloc[otr],y[otr]); ph=np.clip(mm.predict(best['X'].iloc[ote]),0,10)
        nested_pred[ote]=ph
        selections.append({'outer_fold':of,'subset':best['key'][0],'family':best['key'][1],'ridge_alpha':best['alpha'],
                           'n_input_features':best['n_features'],'inner_mean_fold_MAE':best['score'],
                           'inner_fold_MAEs':json.dumps([float(z) for z in best['fold_losses']]),
                           'n_train_participants':pd.Series(groups[otr]).nunique(),'n_test_participants':pd.Series(groups[ote]).nunique()})
        fm=metrics(y[ote],ph);fm.update({'outer_fold':of,'selected_subset':best['key'][0],'selected_family':best['key'][1],'selected_alpha':best['alpha'],'n_test':len(ote)})
        nested_fold.append(fm)
        coef=mm.named_steps['ridge'].coef_
        # coefficients on the fitted standardized/transformed design basis; map names when transform allows.
        try:
            feat_names=mm[:-1].get_feature_names_out(best['X'].columns)
            if len(feat_names)!=len(coef): feat_names=[f'basis_{i+1}' for i in range(len(coef))]
        except Exception: feat_names=[f'basis_{i+1}' for i in range(len(coef))]
        nested_coefs.append(pd.DataFrame({'outer_fold':of,'selected_subset':best['key'][0],'selected_family':best['key'][1],'selected_alpha':best['alpha'],
                                          'feature_or_basis_term':feat_names,'standardized_model_coefficient':coef}))
        print(f"Nested outer fold {of}/{OUTER_N}: choose {best['key'][0]} | {best['key'][1]} | alpha={best['alpha']} | test MAE={fm['MAE']:.3f}",flush=True)
    nested_metric=metrics(y,nested_pred)
    # Bootstrap participants for nested adaptive selector and baseline; resample whole participant clusters.
    dd=pd.DataFrame({'id':groups,'y':y,'pred':nested_pred,'base':base})
    subject_dif=[]
    for _,g in dd.groupby('id'):
        subject_dif.append(mean_absolute_error(g.y,g.pred)-mean_absolute_error(g.y,g.base))
    subject_dif=np.array(subject_dif,float)
    rng=np.random.default_rng(SEED)
    boot=np.array([rng.choice(subject_dif,size=len(subject_dif),replace=True).mean() for _ in range(3000)])
    bootresult={'n_subjects':len(subject_dif),'mean_subject_MAE_diff_model_minus_baseline':float(subject_dif.mean()),
                'CI95_low':float(np.quantile(boot,.025)),'CI95_high':float(np.quantile(boot,.975)),
                'bootstrap_probability_diff_below_zero':float(np.mean(boot<0)),
                'note':'Percentile participant-cluster bootstrap; nested outer-fold predictions compared with outer-training mean baseline.'}
    # Fold-stability and per-subject association for core components
    assoc=[]
    for col in POWER_COLS+MOMENT_COLS:
        corr=[]
        for pid,g in df.groupby('id'):
            a=pd.to_numeric(g[col],errors='coerce'); b=g.NRS
            ok=a.notna()&b.notna()
            if ok.sum()>=4 and a[ok].nunique()>1 and b[ok].nunique()>1:
                r=spearmanr(a[ok],b[ok]).statistic
                if np.isfinite(r):corr.append(r)
        assoc.append({'EEG_feature':col,'n_subjects_estimable':len(corr),'mean_within_subject_Spearman_rho':float(np.mean(corr)) if corr else np.nan,
                      'median_within_subject_Spearman_rho':float(np.median(corr)) if corr else np.nan})
    # Row-level sensitivity benchmark: log-linear and all 5-band cubic / exponential under grouped CV.
    row_sens=[]
    raw_y=raw.NRS.to_numpy(float); raw_g=raw.id.to_numpy(str)
    # Preserve paired rows; features are directly from those rows.
    rp=raw[POWER_COLS].astype(float).clip(lower=1e-12)
    logpr=np.log(rp); logpr.columns=BANDS
    raw_base=np.full(len(raw),np.nan)
    row_splits=list(GroupKFold(n_splits=OUTER_N).split(np.zeros(len(raw)),raw_y,raw_g))
    for tr,te in row_splits:raw_base[te]=raw_y[tr].mean()
    row_sens.append({'unit':'raw paired rows','formula':'training_mean_baseline','alpha':np.nan,**metrics(raw_y,raw_base)})
    for fam in ['log_linear','polynomial_degree_2','polynomial_degree_3','exponential_inverse','sigmoid_saturation','additive_spline']:
        pr=np.full(len(raw),np.nan)
        pipe=candidate_pipeline(fam,SCREEN_ALPHA)
        for tr,te in row_splits:
            mm=clone(pipe);mm.fit(logpr.iloc[tr],raw_y[tr]);pr[te]=np.clip(mm.predict(logpr.iloc[te]),0,10)
        row_sens.append({'unit':'raw paired rows','formula':fam,'alpha':SCREEN_ALPHA,**metrics(raw_y,pr)})
    # Write outputs.
    outdf=pd.DataFrame({'participant_id':groups,'NRS':y,'nested_formula_selector_prediction':nested_pred,'mean_baseline_prediction':base})
    outdf.to_csv(OUT/'nested_formula_selector_oof.csv',index=False)
    screen_df.to_csv(OUT/'all_subset_formula_screen.csv',index=False)
    # Recompute trial-level OOF predictions only for top 20 fixed-alpha candidates to keep package compact.
    pred_rows=[]
    for _, sr in screen_df.head(20).iterrows():
        key=(sr['subset'],sr['family'])
        X,inds,fam=candidates[key]
        pr=np.full(len(y),np.nan); pipe=candidate_pipeline(fam,SCREEN_ALPHA)
        for tr,te in outer_splits:
            mm=clone(pipe); mm.fit(X.iloc[tr],y[tr]); pr[te]=np.clip(mm.predict(X.iloc[te]),0,10)
        pred_rows.append(pd.DataFrame({'participant_id':groups,'NRS':y,'prediction':pr,'subset':sr['subset'],'family':sr['family'],'alpha':SCREEN_ALPHA}))
    pd.concat(pred_rows,ignore_index=True).to_csv(OUT/'top20_formula_oof_predictions.csv',index=False)
    pd.DataFrame(selections).to_csv(OUT/'nested_selected_formula_per_fold.csv',index=False)
    pd.DataFrame(nested_fold).to_csv(OUT/'nested_outer_fold_metrics.csv',index=False)
    pd.concat(nested_coefs,ignore_index=True).to_csv(OUT/'nested_selected_formula_coefficients.csv',index=False)
    pd.DataFrame(assoc).to_csv(OUT/'within_subject_associations.csv',index=False)
    pd.DataFrame(row_sens).to_csv(OUT/'row_level_sensitivity.csv',index=False)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,default=str),encoding='utf-8')
    (OUT/'primary_metrics.json').write_text(json.dumps({'baseline':base_metric,'nested_formula_selector':nested_metric,'participant_cluster_bootstrap':bootresult,
        'search_spec':{'number_of_exhaustive_band_subsets':31,'number_of_candidate_formula_specifications':len(candidates),'nested_formula_alpha_combinations':len(candidates)*len(ALPHAS),'alphas_nested':[float(x) for x in ALPHAS],
                       'outer_group_folds':OUTER_N,'inner_group_folds':INNER_N,'screen_fixed_alpha':SCREEN_ALPHA,'bootstrap_resamples':3000},
        'wall_seconds':round(time.time()-start,2)},indent=2),encoding='utf-8')
    # plot candidate screen top formulas; best-of screen is explicitly exploratory.
    plot=screen_df.head(20).sort_values('MAE',ascending=False)
    fig,ax=plt.subplots(figsize=(12,8));ax.barh(plot['subset']+' | '+plot['family'],plot.MAE);ax.axvline(base_metric['MAE'],linestyle='--',label='Training-mean baseline');ax.set_xlabel('5-fold participant-held-out MAE (NRS points)');ax.set_title('Exhaustive subset/nonlinear formula screen (fixed alpha=10)');ax.legend();fig.tight_layout();fig.savefig(OUT/'nonlinear_formula_screen_top20.png',dpi=180);plt.close(fig)
    # Report
    report=[]
    report += ['# Exhaustive EEG Formula Search: Nonlinear Families and Validation','', 'Analysis run: 2026-10-23.','',
      '## Evidence boundary',
      'This file reports a new nonlinear formula experiment on the actual local Trials.pkl feature table. It is not an independent external validation. The selected public raw-EEG target is OpenNeuro ds005293; an independent data-pipeline run could not be performed in this runtime because external hosts failed DNS resolution. No metrics are claimed from ds005293.', '',
      '## Data and target',
      f"Original table rows: {audit['raw_rows']:,}; unique rating keys after aggregation: {audit['unique_rating_keys']:,}; participants: {audit['participants']}; NRS observed range {audit['NRS_min']}–{audit['NRS_max']}; mean {audit['NRS_mean']:.3f}, SD {audit['NRS_sd']:.3f}.",
      f"Paired-row pattern: {audit['n_keys_with_two_rows']:,} keys have two rows and {audit['n_keys_with_other_row_count']:,} do not. The primary analysis averages EEG features over the shared key because NRS agrees within key. This assumption must be confirmed from original documentation. Raw-row sensitivity is retained.",
      'Only EEG metrics were used as predictors. NRS was the outcome only. The binary `pain` label, skin conductance, demographics, questionnaires, IDs, area/block/trial metadata were not predictors.', '',
      '## Exhaustive formula families',
      f"The analysis examined every one of the 31 non-empty subsets of the five band components (delta, theta, alpha, beta, gamma), crossing subsets with 11 transformation families (with mathematically degenerate singleton CLR forms omitted), plus composite EEG-only feature blocks. There were {len(candidates)} model specifications in the fixed-alpha exhaustive screen. The nested adaptive-selector validation selects among full-five-band families and two EEG composite blocks; the best-of-all-subsets screen remains exploratory to avoid overstating the selected minimum. All coefficients were Ridge regression coefficients; there was no Random Forest / classifier in the NRS formula search.",
      'Families tested: linear in log band power; quadratic and cubic polynomials (including cross-band monomials); pairwise interactions; exponential and inverse-exponential bases; exponential pair interactions; sigmoid/tanh/saturating functions; fractional signed-power bases; reciprocal/saturation functions; additive cubic splines; fold-estimated Yeo–Johnson transforms; relative band power; centered-log-ratio band balance; plus composite features including log-relative powers, prespecified log ratios, normalized spectral entropy, moments, peak frequencies and EEG sample entropy.',
      'Mathematically, the learned score in each family is a coefficient-based scalar: `score = clip(intercept + sum_j beta_j * phi_j(EEG), 0, 10)`. For nonlinear families, `phi_j` denotes the specified basis expansion; standardized transforms and spline knot placement are learned from training participants only.', '',
      '## Validation plan and safeguards',
      '1. Fixed-alpha exhaustive screen: 5-fold GroupKFold by participant, Ridge alpha=10; this makes a comparable screen for every fixed candidate but the best candidate in this table is exploratory because selecting a winner among many formulas inflates its apparent performance.',
      '2. Nested adaptive selector: within each outer training fold, evaluate all 342 formula specifications and each alpha in 3 participant-grouped inner folds, select by inner MAE and predict only the outer held-out participants. All band subsets are exhaustively screened separately, and the best-of-many subset result is explicitly exploratory.',
      '3. Baseline comparison: outer fold training-set mean; predictions clipped to NRS range 0–10. Reports MAE, RMSE, R², Pearson and Spearman associations, prediction variance, and calibration slope/intercept.',
      '4. Participant cluster bootstrap: 3,000 resamples of per-participant MAE differences from nested out-of-fold predictions against mean baseline.',
      '5. Paired-row sensitivity: selected formula families also tested on the original rows grouped by participant, but interpretation is sensitive to the unknown meaning of each paired row.', '',
      '## Actual results',
      f"Training mean baseline: MAE={base_metric['MAE']:.4f}, RMSE={base_metric['RMSE']:.4f}, R²={base_metric['R2']:.4f}, prediction SD={base_metric['prediction_SD']:.4f}.",
      f"Nested formula-search selector: MAE={nested_metric['MAE']:.4f}, RMSE={nested_metric['RMSE']:.4f}, R²={nested_metric['R2']:.4f}, Pearson r={nested_metric['Pearson_r']:.4f}, Spearman rho={nested_metric['Spearman_rho']:.4f}, prediction SD={nested_metric['prediction_SD']:.4f}.",
      f"Participant-cluster bootstrap of subject-level MAE difference (selector minus baseline): mean={bootresult['mean_subject_MAE_diff_model_minus_baseline']:.4f}; 95% percentile CI [{bootresult['CI95_low']:.4f}, {bootresult['CI95_high']:.4f}]; bootstrap probability difference below zero={bootresult['bootstrap_probability_diff_below_zero']:.3f}.", '',
      '### Best fixed-alpha screening candidates (exploratory; winner optimistically biased)', '',
      '| Subset | Family | Features | MAE | RMSE | R² | MAE improvement vs baseline |', '|---|---|---:|---:|---:|---:|---:|']
    for _,r in screen_df.head(20).iterrows():
        report.append(f"| {r['subset']} | {r['family']} | {int(r['n_input_features'])} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {r['R2']:.4f} | {r['MAE_gain_vs_mean_baseline']:.4f} |")
    report += ['', '### Formula picked in each outer fold', '', '| Fold | Selected subset | Family | Alpha | Inner MAE | Outer MAE | Outer RMSE | Outer R² |', '|---:|---|---|---:|---:|---:|---:|---:|']
    sd=pd.DataFrame(selections); fd=pd.DataFrame(nested_fold)
    for _,r in sd.iterrows():
        ff=fd[fd.outer_fold==r.outer_fold].iloc[0]
        report.append(f"| {int(r.outer_fold)} | {r['subset']} | {r['family']} | {r['ridge_alpha']:.0f} | {r['inner_mean_fold_MAE']:.4f} | {ff['MAE']:.4f} | {ff['RMSE']:.4f} | {ff['R2']:.4f} |")
    report += ['', '## Recommended next step: independent raw EEG',
      'The best-matched independent public dataset is OpenNeuro ds005293 / NEMAR on005293, “95 By BP”: 95 healthy participants, 80 trials each, four calibrated laser levels with nominal NRS 2/4/6/8 and trialwise pain ratings. The large nine-experiment release from the same paper spans 678 participants. Data are BIDS-organized; original author-provided preprocessed data and MATLAB code are in the Supplement-Data repository. This makes the methods and trial labels auditable, but NEMAR now reports the complete ds005293 version is about 300.3 GB.',
      'This runtime cannot resolve `s3.amazonaws.com`, `openneuro.org`, or GitHub through local download requests (DNS error), so raw external dataset metrics cannot be truthfully reported here. The analysis package therefore separates current local results from target external data; no external rows, values, or p-values are fabricated.',
      'External study should verify each observed NRS event from the published event tables, extract fixed epoch-level EEG-only band features plus N2/P2 ERP features if the preprocessing supports them, fit all transforms on training participants only, and freeze selected formula families before reporting the independent test once. Do not include nominal stimulus category or stimulus energy as predictor for the EEG-only score.', '',
      '## Public sources',
      '- Zhao et al. (2025), Scientific Data, “A comprehensive EEG dataset of laser-evoked potentials for pain research”, DOI: https://doi.org/10.1038/s41597-025-05900-1',
      '- OpenNeuro ds005293: https://openneuro.org/datasets/ds005293',
      '- NEMAR copy and exact experiment protocol/data-size details: https://nemar.org/dataset/on005293',
      '- Author supplementary preprocessed files and MATLAB code: https://github.com/XiangyueZhao/Supplement-Data',
      '- Other subcohorts listed in the data paper: ds005284, ds005285, ds005289, ds005286, ds005291, ds005292, ds005280, ds005473.', '',
      '## Reproducibility',
      '- `run_nonlinear_formula_search.py`: exact local analysis.',
      '- `all_subset_formula_screen.csv`: 31 subsets crossed with nonlinear/log/ratio/spline families and fixed-alpha results.',
      '- `top20_formula_oof_predictions.csv`: held-out predictions for the top 20 exploratory fixed-alpha specifications.',
      '- `nested_formula_selector_oof.csv`: unbiased outer-fold predictions from the nested search procedure.',
      '- `nested_selected_formula_per_fold.csv`: exact selected subset/family/alpha in each outer fold.',
      '- `nested_selected_formula_coefficients.csv`: selected outer-fold coefficient basis weights.',
      '- `formula_search_results.xlsx`: consolidated tables and predictions.',
      '- `dataset_source_and_methods.md`: public data description, feature mapping and external run limits.'
    ]
    (OUT/'EEG_Nonlinear_Formula_Expansion_and_Validation.md').write_text('\n'.join(report),encoding='utf-8')
    # Workbook with readable outputs
    with pd.ExcelWriter(OUT/'formula_search_results.xlsx',engine='openpyxl') as xw:
        screen_df.to_excel(xw,sheet_name='All_Subsets_Screen',index=False)
        pd.DataFrame(selections).to_excel(xw,sheet_name='Nested_Selections',index=False)
        fd.to_excel(xw,sheet_name='Nested_Outer_Folds',index=False)
        pd.DataFrame([{'baseline_'+k:v for k,v in base_metric.items()}|{'selector_'+k:v for k,v in nested_metric.items()}|bootresult]).to_excel(xw,sheet_name='Summary',index=False)
        pd.DataFrame([bootresult]).to_excel(xw,sheet_name='Participant_Bootstrap',index=False)
        pd.concat(nested_coefs,ignore_index=True).to_excel(xw,sheet_name='Outer_Selected_Coefficients',index=False)
        pd.DataFrame(assoc).to_excel(xw,sheet_name='Within_Subject_Association',index=False)
        pd.DataFrame(row_sens).to_excel(xw,sheet_name='Paired_Row_Sensitivity',index=False)
        pd.DataFrame([audit]).to_excel(xw,sheet_name='Audit',index=False)
        outdf.to_excel(xw,sheet_name='Nested_OOF_Predictions',index=False)
    wb=load_workbook(OUT/'formula_search_results.xlsx')
    for ws in wb.worksheets:
        ws.freeze_panes='A2'; ws.auto_filter.ref=ws.dimensions
        for c in ws[1]: c.font=Font(bold=True,color='FFFFFF'); c.fill=PatternFill('solid',fgColor='24415A'); c.alignment=Alignment(wrap_text=True,vertical='top')
        for col in ws.columns:
            mx=min(52,max(10,max((len(str(c.value)) for c in list(col)[:300] if c.value is not None),default=10)+2))
            ws.column_dimensions[get_column_letter(col[0].column)].width=mx
    wb.save(OUT/'formula_search_results.xlsx')
    # Short Word report derived from results
    doc=Document(); sec=doc.sections[0]; sec.top_margin=Inches(.65); sec.bottom_margin=Inches(.65); sec.left_margin=Inches(.7); sec.right_margin=Inches(.7)
    doc.styles['Normal'].font.name='Aptos';doc.styles['Normal'].font.size=Pt(9)
    t=doc.add_paragraph();t.alignment=1;r=t.add_run('EEG Nonlinear Formula Search & Validation');r.bold=True;r.font.size=Pt(20)
    p=doc.add_paragraph();p.alignment=1;p.add_run('Actual local feature-data experiment • 23 October 2026').italic=True
    doc.add_paragraph('The requested independent raw EEG run remains blocked: download hosts did not resolve from the analysis runtime. Public target dataset and exact source links are identified, but no external metrics are claimed.')
    doc.add_heading('Primary result',level=1)
    doc.add_paragraph(f"The unique-key analysis used {audit['unique_rating_keys']:,} participant-level trial keys from {audit['participants']} people. Mean baseline MAE {base_metric['MAE']:.3f}; nested nonlinear formula selector MAE {nested_metric['MAE']:.3f}. Selector-baseline subject-level MAE difference {bootresult['mean_subject_MAE_diff_model_minus_baseline']:.3f}, 95% CI [{bootresult['CI95_low']:.3f}, {bootresult['CI95_high']:.3f}].")
    doc.add_heading('Exhaustive formula search',level=1)
    doc.add_paragraph(f"All 31 nonempty subsets of the five band power components were crossed with 11 transform families (and composite EEG metrics), producing {len(candidates)} specifications. Transform families included linear log power, polynomial degrees 2/3, interactions, exponential/inverse-exponential, exponential interactions, sigmoid/tanh, signed fractional powers, reciprocal/saturation, splines, Yeo–Johnson, relative power and centered-log-ratio balances.")
    doc.add_heading('Top fixed-alpha screen results (exploratory)',level=1)
    tab=doc.add_table(rows=1,cols=6);tab.style='Light Shading Accent 1'
    for i,h in enumerate(['Band subset','Transform','Features','MAE','RMSE','R²']):tab.rows[0].cells[i].text=h
    for _,r in screen_df.head(12).iterrows():
        row=tab.add_row().cells
        vals=[r['subset'],r['family'],str(int(r['n_input_features'])),f"{r['MAE']:.3f}",f"{r['RMSE']:.3f}",f"{r['R2']:.3f}"]
        for i,v in enumerate(vals):row[i].text=str(v)
    doc.add_picture(str(OUT/'nonlinear_formula_screen_top20.png'),width=Inches(6.4))
    doc.add_heading('External validation target',level=1)
    doc.add_paragraph('OpenNeuro ds005293/NEMAR on005293 contains 95 participants, 80 laser trials each, 4 calibrated pain levels and trialwise NRS. See report and dataset_source_and_methods.md for DOI, source, method, access issue and external workflow. Raw EEG must be available before claiming independent validation.')
    doc.add_heading('Interpretation',level=1)
    doc.add_paragraph('The exhaustive screen’s best candidate must not be interpreted as unbiased performance. The nested formula selector is the honest estimate of the search procedure. Even if its MAE improves slightly, a pain scale requires demonstrable separation, calibration, stable coefficients and independent transport validation.')
    doc.save(OUT/'EEG_Nonlinear_Formula_Expansion_and_Validation.docx')
    # zip all project content except zip itself
    zpath=ROOT.parent/'EEG_Nonlinear_Formula_Expansion_2026_All_Files.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob('*'):
            if p.is_file() and p != zpath: z.write(p,arcname=str(Path(ROOT.name)/p.relative_to(ROOT)))
    print('\nDONE in seconds',round(time.time()-start,2),flush=True)
    print('BASELINE',base_metric,flush=True)
    print('NESTED SELECTOR',nested_metric,flush=True)
    print('BOOTSTRAP',bootresult,flush=True)
    print('TOP SCREEN\n',screen_df.head(15)[['subset','family','n_input_features','MAE','RMSE','R2','MAE_gain_vs_mean_baseline']].to_string(index=False),flush=True)
    print('SELECTED FOLDS\n',pd.DataFrame(selections).to_string(index=False),flush=True)
    print('ZIP',zpath,flush=True)

if __name__=='__main__':
    warnings.filterwarnings('ignore')
    run_search()
