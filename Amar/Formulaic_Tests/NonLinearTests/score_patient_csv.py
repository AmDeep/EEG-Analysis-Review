#!/usr/bin/env python3
"""Research-only EEG pain score for participant trial-feature CSVs.

Input requirements:
- One row per documented trial/epoch, with EEG_delta, EEG_theta, EEG_alpha,
  EEG_beta, EEG_gamma band-power values computed using the same preprocessing and
  units as the source dataset.
- If a `pain` column is present, rows with pain != 1 are removed. If it is absent,
  pass --already-pain-epochs only if the file contains pain-response epochs only.
- A patient identifier column is required (default: `id`). It groups observations
  but is never a predictor.

Outputs both the continuous prediction of NRS_avg4wk and a direct three-class
multinomial-formula label. Both are experimental research outputs, not clinical advice.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

EPS=1e-6
BANDS=['EEG_delta','EEG_theta','EEG_alpha','EEG_beta','EEG_gamma']
BAND_NAMES=['delta','theta','alpha','beta','gamma']

def label_from_score(x):
    if x < 4: return 'Low/mild (predicted NRS < 4)'
    if x < 7: return 'Moderate (predicted NRS 4-<7)'
    return 'High/severe (predicted NRS >= 7)'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True,help='CSV of per-trial EEG bandpower features')
    ap.add_argument('--output',default='eeg_pain_scores.csv',help='Output CSV path')
    ap.add_argument('--id-column',default='id',help='Column used to group trials by person')
    ap.add_argument('--already-pain-epochs',action='store_true',help='Assert input contains only pain-response epochs if no pain column is supplied')
    ap.add_argument('--models-dir',default=str(Path(__file__).resolve().parent),help='Directory holding the two fitted .joblib research prototype models')
    args=ap.parse_args()
    raw=pd.read_csv(args.input)
    need=[args.id_column]+BANDS
    missing=[c for c in need if c not in raw.columns]
    if missing: raise SystemExit(f'Missing required columns: {missing}')
    if 'pain' in raw.columns:
        raw=raw.loc[pd.to_numeric(raw['pain'],errors='coerce').eq(1)].copy()
    elif not args.already_pain_epochs:
        raise SystemExit('No `pain` epoch flag found. Add it or explicitly pass --already-pain-epochs after verifying the rows are pain-response epochs.')
    if raw.empty: raise SystemExit('No pain-epoch records remain after filtering pain==1.')
    for c in BANDS:
        raw[c]=pd.to_numeric(raw[c],errors='coerce')
    if raw[BANDS].isna().all(axis=1).any():
        raw=raw.loc[~raw[BANDS].isna().all(axis=1)].copy()
    if raw.empty: raise SystemExit('No valid EEG band-power feature rows remain.')
    rows=[]
    for ident,grp in raw.groupby(args.id_column,sort=True):
        p=grp[BANDS].astype(float).clip(lower=0)
        lp=np.log(p+EPS)
        feat={}
        for name in BAND_NAMES:
            col='log_'+name
            vals=lp[col if col in lp.columns else BANDS[BAND_NAMES.index(name)]] if False else lp.iloc[:,BAND_NAMES.index(name)]
            feat[f'log_{name}_mean']=float(vals.mean())
            feat[f'log_{name}_sd']=float(vals.std(ddof=1)) if len(vals)>1 else np.nan
        rows.append({'id':ident,'n_pain_epochs':len(grp),**feat})
    agg=pd.DataFrame(rows)
    names=[f'log_{b}_mean' for b in BAND_NAMES]+[f'log_{b}_sd' for b in BAND_NAMES]
    X=agg[names].to_numpy(float)
    mdir=Path(args.models_dir)
    score_model=joblib.load(mdir/'EEG_patient_average_pain_score_prototype.joblib')
    label_model=joblib.load(mdir/'EEG_patient_severity_label_formula_prototype.joblib')
    agg['predicted_NRS_avg4wk_score']=np.clip(score_model.predict(X),0,10)
    agg['regression_threshold_label']=agg.predicted_NRS_avg4wk_score.map(label_from_score)
    # Classifier feature schema stored in prototype JSON; this public model selects means+variability on present run.
    import json
    spec=json.loads((mdir/'direct_label_formula_prototype.json').read_text())
    class_features=spec['features']
    class_X=agg[class_features].to_numpy(float)
    probs=label_model.predict_proba(class_X)
    classes=label_model.named_steps['logit'].classes_
    class_to_label={0:'Low/mild',1:'Moderate',2:'High/severe'}
    for j,cl in enumerate(classes): agg[f'uncalibrated_probability_{class_to_label[int(cl)].replace("/","_").replace("-","_").replace(" ","_").lower()}']=probs[:,j]
    pred_class=classes[np.argmax(probs,axis=1)]
    agg['direct_formula_severity_label']=[class_to_label[int(x)] for x in pred_class]
    agg['validation_status']='Research prototype only - not clinically validated'
    agg['important_caveat']='In held-out subjects, average NRS regression was modest; severity classification and leave-one-diagnosis-out generalization were weak/inconsistent.'
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    agg.to_csv(args.output,index=False)
    print(f'Wrote {len(agg)} participant score row(s) to {args.output}')
    print('Research-only prototype. Do not use for diagnosis, triage or treatment decisions.')

if __name__=='__main__': main()
