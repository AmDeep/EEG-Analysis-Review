#!/usr/bin/env python3
"""Full nested grouped evaluation of all 342 EEG-only formula specifications.

Five outer GroupKFold folds; inside each outer train set, 3-fold GroupKFold
selection across all 342 formula specifications with prespecified Ridge alpha=10.
Ridge-penalty sensitivity is screened separately at alpha {1,10,100}. NRS is
outcome only. This file is separated from the screen for resumability. Run e.g.:
  python run_full_nested_formula_search.py --folds 1 2 3
  python run_full_nested_formula_search.py --folds 4 5
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import run_nonlinear_formula_search as core

ROOT=Path(__file__).resolve().parent
OUT=ROOT

def prepare():
    raw, df, y, groups, candidates, audit, _ = core.build_primary_data()
    splits=list(GroupKFold(n_splits=core.OUTER_N).split(np.zeros(len(y)),y,groups))
    return raw,df,y,groups,candidates,audit,splits

def run_fold(args):
    fold_no, out_idx = args
    t0=time.time()
    # Isolate data in this worker; avoids shared mutable pipeline state.
    raw, df, y, groups, candidates, audit, outer_splits=prepare()
    otr, ote=outer_splits[fold_no-1]
    yin=y[otr]; gin=groups[otr]
    inner_splits=list(GroupKFold(n_splits=core.INNER_N).split(np.zeros(len(otr)),yin,gin))
    best={'score':float('inf')}
    ordered=list(candidates.items())
    for ci,(key,(X,inds,fam)) in enumerate(ordered,1):
        Xtr=X.iloc[otr]
        alpha=core.SCREEN_ALPHA
        losses=[]
        pipe=core.candidate_pipeline(fam,alpha)
        for itr,iva in inner_splits:
            mm=clone(pipe)
            mm.fit(Xtr.iloc[itr],yin[itr])
            pred=np.clip(mm.predict(Xtr.iloc[iva]),0,10)
            losses.append(mean_absolute_error(yin[iva],pred))
        score=float(np.mean(losses))
        if score<best['score']:
            best={'score':score,'key':key,'X':X,'fam':fam,'alpha':alpha,'losses':losses,'n_features':X.shape[1]}
        if ci%50==0:
            print(f'Fold {fold_no}: tested {ci}/{len(ordered)} formula specifications',flush=True)
    model=core.candidate_pipeline(best['fam'],best['alpha'])
    model.fit(best['X'].iloc[otr],y[otr])
    pred=np.clip(model.predict(best['X'].iloc[ote]),0,10)
    baseline=np.full(len(ote),float(np.mean(y[otr])))
    fold_metrics=core.metrics(y[ote],pred)
    fold_metrics.update({'outer_fold':fold_no,'selected_subset':best['key'][0], 'selected_family':best['key'][1],
                         'selected_alpha':best['alpha'],'inner_mean_MAE':best['score'],'n_test_rows':len(ote),
                         'n_test_participants':int(pd.Series(groups[ote]).nunique()),'elapsed_seconds':round(time.time()-t0,2)})
    pred_df=pd.DataFrame({'participant_id':groups[ote],'NRS':y[ote],'prediction':pred,'mean_baseline_prediction':baseline,'outer_fold':fold_no})
    pred_df.to_csv(OUT/f'nested_fullspace_fold{fold_no}_predictions.csv',index=False)
    (OUT/f'nested_fullspace_fold{fold_no}_selection.json').write_text(json.dumps({
        'outer_fold':fold_no,'selected_subset':best['key'][0],'selected_family':best['key'][1],
        'ridge_alpha':best['alpha'],'n_input_features':best['n_features'],'inner_fold_MAEs':best['losses'],
        'inner_mean_MAE':best['score'],'n_train_participants':int(pd.Series(groups[otr]).nunique()),
        'n_test_participants':int(pd.Series(groups[ote]).nunique()),'elapsed_seconds':round(time.time()-t0,2)},indent=2))
    coef=model.named_steps['ridge'].coef_
    try:
        names=model[:-1].get_feature_names_out(best['X'].columns)
        if len(names)!=len(coef): names=[f'basis_{i+1}' for i in range(len(coef))]
    except Exception:
        names=[f'basis_{i+1}' for i in range(len(coef))]
    pd.DataFrame({'outer_fold':fold_no,'selected_subset':best['key'][0],'selected_family':best['key'][1],
                  'selected_alpha':best['alpha'],'feature_or_basis_term':names,
                  'standardized_model_coefficient':coef}).to_csv(OUT/f'nested_fullspace_fold{fold_no}_coefficients.csv',index=False)
    pd.DataFrame([fold_metrics]).to_csv(OUT/f'nested_fullspace_fold{fold_no}_metrics.csv',index=False)
    print(f"Fold {fold_no} DONE | {best['key'][0]} | {best['key'][1]} | alpha={best['alpha']} | MAE={fold_metrics['MAE']:.4f} | {fold_metrics['elapsed_seconds']} sec",flush=True)
    return fold_no, fold_metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--folds',nargs='+',type=int,required=True,choices=range(1,6))
    ap.add_argument('--workers',type=int,default=min(3,len(set())) or 3)
    args=ap.parse_args()
    todo=[(f, i) for i,f in enumerate(args.folds)]
    nworkers=min(args.workers,len(todo))
    start=time.time()
    print(f'Running full nested 342-specification x 3-alpha search; outer folds={args.folds}; workers={nworkers}',flush=True)
    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        futs=[ex.submit(run_fold,item) for item in todo]
        for fut in as_completed(futs): fut.result()
    print(f'Batch completed in {time.time()-start:.1f}s',flush=True)
if __name__=='__main__': main()
