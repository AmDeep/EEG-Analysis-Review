#!/usr/bin/env python3
"""Aggregate the five resumable outer folds from run_full_nested_formula_search.py."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
ROOT=Path(__file__).resolve().parent
SEED=20261023

def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); sd=float(np.std(p,ddof=1))
    return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),
            'R2':float(r2_score(y,p)), 'Pearson_r':float(pearsonr(y,p).statistic) if sd>1e-12 else None,
            'Spearman_rho':float(spearmanr(y,p).statistic) if sd>1e-12 else None,
            'prediction_SD':sd,'calibration_slope':float(np.polyfit(p,y,1)[0]) if sd>1e-12 else None,
            'calibration_intercept':float(np.polyfit(p,y,1)[1]) if sd>1e-12 else None}
for f in range(1,6):
    for suffix in ['predictions.csv','selection.json','metrics.csv','coefficients.csv']:
        p=ROOT/f'nested_fullspace_fold{f}_{suffix}'
        if not p.exists(): raise FileNotFoundError(p)
preds=pd.concat([pd.read_csv(ROOT/f'nested_fullspace_fold{f}_predictions.csv') for f in range(1,6)],ignore_index=True)
assert len(preds)==11214, f'Expected 11214 rows, got {len(preds)}'
assert preds['NRS'].between(0,10).all()
model=metrics(preds.NRS,preds.prediction)
base=metrics(preds.NRS,preds.mean_baseline_prediction)
subject_dif=[]
for _,g in preds.groupby('participant_id'):
    subject_dif.append(mean_absolute_error(g.NRS,g.prediction)-mean_absolute_error(g.NRS,g.mean_baseline_prediction))
subject_dif=np.asarray(subject_dif,float)
rng=np.random.default_rng(SEED)
boot=np.array([rng.choice(subject_dif,size=len(subject_dif),replace=True).mean() for _ in range(3000)])
bootresult={'n_subjects':len(subject_dif),'mean_subject_MAE_diff_model_minus_baseline':float(subject_dif.mean()),
            'CI95_low':float(np.quantile(boot,.025)),'CI95_high':float(np.quantile(boot,.975)),
            'bootstrap_probability_diff_below_zero':float(np.mean(boot<0)),
            'note':'Percentile participant-cluster bootstrap of per-participant MAE difference, full-space nested outer-fold predictions versus outer-training mean baseline.'}
selections=pd.DataFrame([json.loads((ROOT/f'nested_fullspace_fold{f}_selection.json').read_text()) for f in range(1,6)])
fold_metrics=pd.concat([pd.read_csv(ROOT/f'nested_fullspace_fold{f}_metrics.csv') for f in range(1,6)],ignore_index=True)
coefs=pd.concat([pd.read_csv(ROOT/f'nested_fullspace_fold{f}_coefficients.csv') for f in range(1,6)],ignore_index=True)
preds.to_csv(ROOT/'nested_fullspace_oof_predictions.csv',index=False)
selections.to_csv(ROOT/'nested_fullspace_selected_formula_per_fold.csv',index=False)
fold_metrics.to_csv(ROOT/'nested_fullspace_outer_fold_metrics.csv',index=False)
coefs.to_csv(ROOT/'nested_fullspace_selected_coefficients.csv',index=False)
primary={'baseline':base,'nested_formula_selector':model,'participant_cluster_bootstrap':bootresult,
         'search_spec':{'n_unique_rating_keys':11214,'n_participants':117,'number_of_exhaustive_band_subsets':31,
                        'number_of_candidate_formula_specifications':342,'nested_ridge_alpha_fixed':10.0,
                        'outer_group_folds':5,'inner_group_folds':3,'bootstrap_resamples':3000,
                        'nested_selector':'all 342 formula specifications; alpha fixed at 10; candidate family/subset selected using inner grouped-CV MAE',
                        'external_public_dataset':'OpenNeuro ds005293 / NEMAR on005293 selected but not downloaded/externally scored in this runtime'}}
(ROOT/'fullspace_nested_metrics.json').write_text(json.dumps(primary,indent=2))
# Compact bootstrap CSV for downstream reporting/workbooks.
pd.DataFrame([bootresult]).to_csv(ROOT/'participant_cluster_bootstrap.csv',index=False)
# Canonical primary metrics file, including the top exploratory screen row (which is NOT the unbiased estimate).
best_screen=pd.read_csv(ROOT/'all_subset_formula_screen.csv').sort_values('MAE').iloc[0].to_dict()
primary['exploratory_best_fixed_alpha_screen']={k:(None if pd.isna(v) else (float(v) if isinstance(v,(np.floating,float)) else (int(v) if isinstance(v,(np.integer,int)) else v))) for k,v in best_screen.items()}
(ROOT/'primary_metrics.json').write_text(json.dumps(primary,indent=2))
print('BASELINE',json.dumps(base,indent=2))
print('FULL-SPACE NESTED',json.dumps(model,indent=2))
print('BOOTSTRAP',json.dumps(bootresult,indent=2))
print('FOLD SELECTIONS\n',selections[['outer_fold','selected_subset','selected_family','ridge_alpha','n_input_features','inner_mean_MAE','n_test_participants']].to_string(index=False))
