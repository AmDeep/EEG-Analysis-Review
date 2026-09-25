#!/usr/bin/env python3
"""Participant-held-out continuous EEG regression. See README before interpreting results.

Commands: benchmark, stress, dose, download-gozzi, external, fit, predict, self-check.
No participant IDs, trial order, pain labels, or acquisition metadata enter EEG X.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time, warnings
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.svm import SVR
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
SEED=20260924
FEATURES=['n2_amp','n2_lat','p2_amp','p2_lat','n2p2_amp','gamma_power',
 'alpha_erd_pct','beta_erd_pct','psd_delta','psd_theta','psd_alpha','psd_beta','psd_gamma',
 'plv_Fz-Cz','plv_Cz-Pz','plv_C3-C4','perm_entropy','spectral_entropy',
 'sample_entropy','higuchi_fd','dfa','hjorth_mobility','hjorth_complexity']
VARIABLE_STUDIES=['ds005280','ds005285','ds005292','ds005293','ds005473']
CANDIDATES={
 'ridge':[{'alpha':a} for a in [1,100,10000]],
 'spline_ridge':[{'n_knots':k,'alpha':a} for k,a in [(3,100),(5,100),(5,1000)]],
 'svr':[{'C':c,'gamma':g} for c,g in [(1,.01),(10,.01),(1,.05)]],
 'extra_trees':[{'min_samples_leaf':l,'max_features':m} for l,m in [(5,.8),(20,1.),(50,1.)]],
 'random_forest':[{'min_samples_leaf':l,'max_features':m} for l,m in [(10,.8),(40,1.)]],
 'hist_gradient_boosting':[{'max_leaf_nodes':l,'l2_regularization':r,'max_iter':n} for l,r,n in [(7,10,200),(15,10,200),(15,100,400),(31,100,200)]],
 'mlp':[{'hidden_layer_sizes':h,'alpha':a} for h,a in [((64,32),1.),((32,),10.)]]}

class RobustFeatures(BaseEstimator,TransformerMixin):
    """Train-only winsorization; signed-log EEG powers/ERD and power ratios."""
    def __init__(self, columns=None): self.columns=columns
    def _map(self,X):
        x=np.asarray(X,dtype=float).copy(); x[~np.isfinite(x)]=np.nan
        cols=list(self.columns or [])
        for j,c in enumerate(cols):
            if c.startswith('psd_') or c in ('gamma_power','alpha_erd_pct','beta_erd_pct'):
                x[:,j]=np.sign(x[:,j])*np.log1p(np.abs(x[:,j]))
        # Fixed, label-free ratios; no fit on held-out subjects.
        if all(c in cols for c in ['psd_delta','psd_theta','psd_alpha','psd_beta','psd_gamma']):
            raw=np.asarray(X,dtype=float)
            p=np.maximum(raw[:,[cols.index('psd_'+b) for b in ['delta','theta','alpha','beta','gamma']]],1e-12)
            rel=p/np.sum(p,axis=1,keepdims=True)
            x=np.c_[x,rel,np.log(p[:,4]/p[:,2]),np.log(p[:,3]/p[:,2])]
        return x
    def fit(self,X,y=None):
        x=self._map(X)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            self.lo_=np.nanquantile(x,.005,axis=0);self.hi_=np.nanquantile(x,.995,axis=0)
        self.lo_=np.nan_to_num(self.lo_,nan=0);self.hi_=np.nan_to_num(self.hi_,nan=0)
        return self
    def transform(self,X): return np.clip(self._map(X),self.lo_,self.hi_)

def make_model(family,params,columns):
    p=dict(params)
    pre=[RobustFeatures(columns),SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()]
    if family=='ridge': est=Ridge(**p)
    elif family=='spline_ridge':
        k=p.pop('n_knots');pre.append(SplineTransformer(n_knots=k,degree=3,knots='quantile',extrapolation='linear'));est=Ridge(**p)
    elif family=='svr': est=SVR(epsilon=.1,cache_size=512,**p)
    elif family=='extra_trees':est=ExtraTreesRegressor(n_estimators=200,n_jobs=1,random_state=SEED,**p)
    elif family=='random_forest':est=RandomForestRegressor(n_estimators=160,n_jobs=1,random_state=SEED,**p)
    elif family=='hist_gradient_boosting':est=HistGradientBoostingRegressor(learning_rate=.05,min_samples_leaf=30,early_stopping=False,random_state=SEED,**p)
    elif family=='mlp':est=MLPRegressor(max_iter=150,early_stopping=False,random_state=SEED,batch_size=256,learning_rate_init=.001,**p)
    else:raise ValueError(family)
    return TransformedTargetRegressor(regressor=make_pipeline(*pre,est),transformer=StandardScaler())

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(p)
def read_json(path):return json.loads(Path(path).read_text())
def finite(v):return float(v) if np.isfinite(v) else None

def load_zhao(path):
    d=pd.read_csv(path)
    needed=['dataset','subject','epoch','rating','laser_power']+FEATURES
    if not set(needed)<=set(d):raise ValueError('Missing Zhao columns')
    if d.duplicated(['dataset','subject','epoch']).any():raise ValueError('Duplicate trial keys: resolve session/run identity first')
    d['source_row']=np.arange(len(d));d['group']=d.dataset.astype(str)+'/'+d.subject.astype(str)
    d.loc[~d.rating.between(0,10),'rating']=np.nan
    # Quarantine whole subjects with identical feature vectors: observed cross-ID
    # copies and repeated EEG carrying contradictory labels. Do not guess truth.
    suspect=d.loc[d[FEATURES].duplicated(keep=False),'group'].unique()
    if len(suspect):
        print('Quarantined duplicate-feature subjects:', ', '.join(sorted(suspect)),flush=True)
        d=d[~d.group.isin(suspect)].copy()
    return d

def split_subjects(d):
    """20% sealed test subjects per study; dev subjects assigned 3 group folds.
    Assignment depends on IDs only, never trial counts, EEG, or labels.
    """
    out=d.copy();out['split']='dev';out['fold']=-1
    for ds,g in d.groupby('dataset',sort=True):
        ids=sorted(g.group.unique(),key=lambda x:hashlib.sha256(f'{SEED}/{x}'.encode()).hexdigest())
        ntest=max(2,int(np.ceil(.2*len(ids))))
        if len(ids)-ntest<3:raise ValueError('Need >=5 participant groups per dataset')
        out.loc[out.group.isin(ids[:ntest]),'split']='test'
        for i,sub in enumerate(ids[ntest:]):out.loc[out.group==sub,'fold']=i%3
    for fold in range(3):
        a=set(out.loc[(out.split=='dev')&(out.fold!=fold),'group']);b=set(out.loc[(out.split=='dev')&(out.fold==fold),'group'])
        assert a.isdisjoint(b)
    assert set(out.loc[out.split=='dev','group']).isdisjoint(out.loc[out.split=='test','group'])
    return out

def task_data(d,task):
    if task=='stimulus':return d[d.dataset.isin(VARIABLE_STUDIES)&d.laser_power.notna()].copy(),'laser_power',FEATURES
    if task=='pain':return d[d.rating.notna()].copy(),'rating',FEATURES
    raise ValueError(task)

def macro_mae(d,y,p):
    e=pd.DataFrame({'dataset':np.asarray(d.dataset),'group':np.asarray(d.group),'ae':np.abs(np.asarray(y)-np.asarray(p))})
    return float(e.groupby(['dataset','group']).ae.mean().groupby('dataset').mean().mean())

def metrics(d,y,p,bootstrap=0):
    y=np.asarray(y);p=np.asarray(p)
    z=pd.DataFrame({'ds':np.asarray(d.dataset),'g':np.asarray(d.group),'y':y,'p':p,'ae':np.abs(y-p)})
    wc=z[['y','p']]-z.groupby('g')[['y','p']].transform('mean')
    def corr(a,b):return finite(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else None
    r={'n':len(y),'subjects':int(d.group.nunique()),'mae':float(mean_absolute_error(y,p)),
       'rmse':float(np.sqrt(mean_squared_error(y,p))),'r2':finite(r2_score(y,p)),
       'pearson_r':corr(y,p),'spearman_r':finite(spearmanr(y,p).statistic) if np.std(p)>1e-12 else None,
       'within_subject_r':corr(wc.y,wc.p),'macro_mae':macro_mae(d,y,p)}
    if bootstrap:
        # Stratified participant cluster bootstrap; each study and subject equal weight.
        means=z.groupby(['ds','g']).ae.mean()
        rng=np.random.default_rng(SEED);samples=np.zeros(bootstrap)
        for ds in means.index.get_level_values(0).unique():
            a=means.loc[ds].to_numpy();samples+=rng.choice(a,(bootstrap,len(a)),replace=True).mean(axis=1)
        samples/=means.index.get_level_values(0).nunique()
        r['macro_mae_ci95']=[float(x) for x in np.quantile(samples,[.025,.975])]
    return r

def fit_predict(family,params,columns,train,test,target):
    with threadpool_limits(1),warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always',ConvergenceWarning)
        m=make_model(family,params,columns)
        m.fit(train[columns],train[target]);p=m.predict(test[columns])
    return p,any(issubclass(w.category,ConvergenceWarning) for w in ws)

def evaluate_candidate(family,params,dev,target,columns):
    start=time.time();pred=np.full(len(dev),np.nan);conv=False
    for k in range(3):
        a=dev.fold!=k;b=~a
        p,w=fit_predict(family,params,columns,dev[a],dev[b],target);pred[np.where(b)[0]]=p;conv|=w
    ans={'family':family,'params':params,'cv':metrics(dev,dev[target],pred),'seconds':round(time.time()-start,2),'convergence_warning':conv}
    print(f"{target}: {family} {params} CV macroMAE={ans['cv']['macro_mae']:.5f}",flush=True)
    return ans

def run_benchmark(args):
    d=split_subjects(load_zhao(args.data));out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    path=out/'results.json'
    if path.exists():
        r=read_json(path)
        if r['data_sha256']!=sha(args.data):raise ValueError('Data differ from cached run; choose a new --out')
    else:r={'seed':SEED,'data_sha256':sha(args.data),'code_sha256':sha(__file__),'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'tasks':{}}
    prediction_blocks=[]
    # Persist IDs so every score is auditable, including all held-out predictions.
    for task in args.tasks:
        td,target,cols=task_data(d,task);dev=td[td.split=='dev'].copy();test=td[td.split=='test'].copy()
        tr=r['tasks'].setdefault(task,{'target':target,'features':cols,'candidates':[]})
        completed={(x['family'],json.dumps(x['params'],sort_keys=True)) for x in tr['candidates']}
        todo=[(f,p) for f,ps in CANDIDATES.items() for p in ps if (f,json.dumps(p,sort_keys=True)) not in completed]
        # Store each finished family; parallelism uses independent local fits, not agents.
        for family in CANDIDATES:
            jobs=[(f,p) for f,p in todo if f==family]
            if jobs:
                tr['candidates'].extend(Parallel(n_jobs=args.jobs)(delayed(evaluate_candidate)(f,p,dev,target,cols) for f,p in jobs));write_json(path,r)
        bests=sorted([min([x for x in tr['candidates'] if x['family']==f],key=lambda x:x['cv']['macro_mae']) for f in CANDIDATES],key=lambda x:x['cv']['macro_mae'])
        tr['family_ranking']=bests;tr['finalists']=[{'family':x['family'],'params':x['params']} for x in bests[:2]]
        tr['train_n']=len(dev);tr['test_n']=len(test);tr['train_subjects']=dev.group.nunique();tr['test_subjects']=test.group.nunique()
        tr['test']={};tr['by_study']={}
        predcols=test[['source_row','dataset','subject','epoch','group','split','fold']].copy();predcols['task']=task;predcols['y']=test[target]
        # Benchmark a per-study training mean. Dataset ID is used ONLY by this baseline.
        means=dev.groupby('dataset')[target].mean();base=test.dataset.map(means).fillna(dev[target].mean()).to_numpy()
        preds={'study_mean':base}
        for x in tr['finalists']:
            p,w=fit_predict(x['family'],x['params'],cols,dev,test,target);preds[x['family']]=p
            print(f"TEST {task} {x['family']} {metrics(test,test[target],p)}",flush=True)
        preds['equal_blend']=np.mean([preds[x['family']] for x in tr['finalists']],axis=0)
        # Blend is a fixed, supplementary average; does not change selected family ranking.
        for name,p in preds.items():
            tr['test'][name]=metrics(test,test[target],p,2000)
            predcols[name]=p
            tr['by_study'][name]={ds:metrics(g,g[target],p[test.dataset.to_numpy()==ds],1000) for ds,g in test.groupby('dataset')}
        # Paired uncertainty of first-vs-second selected family; lower is better.
        a,b=[x['family'] for x in tr['finalists']]
        delta=pd.DataFrame({'ds':test.dataset.to_numpy(),'g':test.group.to_numpy(),'v':abs(test[target].to_numpy()-preds[a])-abs(test[target].to_numpy()-preds[b])}).groupby(['ds','g']).v.mean()
        rng=np.random.default_rng(SEED);samples=np.zeros(2000)
        for ds in delta.index.get_level_values(0).unique():
            v=delta.loc[ds].to_numpy();samples+=rng.choice(v,(2000,len(v))).mean(axis=1)
        samples/=delta.index.get_level_values(0).nunique()
        tr['paired_mae_difference']={'contrast':f'{a} minus {b}','value':float(delta.groupby(level=0).mean().mean()),'ci95':np.quantile(samples,[.025,.975]).tolist()}
        prediction_blocks.append(predcols);write_json(path,r)
    if prediction_blocks:
        pp=out/'predictions.csv.gz'
        if pp.exists():
            old=pd.read_csv(pp);prediction_blocks.insert(0,old[~old.task.isin(args.tasks)])
        pd.concat(prediction_blocks,ignore_index=True).to_csv(pp,index=False,compression='gzip')
    # Fold assignments for all rows are cheap to reproduce from the stable hash in split_subjects.
    print('Saved',path,flush=True)

def dose_analysis(d):
    valid=d[d.rating.notna()].copy();res={};rng=np.random.default_rng(SEED)
    for ds,g in valid.groupby('dataset'):
        stats=[];curves=[]
        for sub,h in g.groupby('group'):
            x=h.laser_power.to_numpy();y=h.rating.to_numpy();xc=x-x.mean();yc=y-y.mean()
            den=float(xc@xc)
            if den>1e-10:stats.append((float(xc@yc),den,float((xc@yc)/den)))
        if not stats:res[ds]={'estimable':False,'reason':'Only one stimulus intensity within every participant.'};continue
        a=np.array(stats);inds=rng.integers(0,len(a),(3000,len(a)));s=a[inds]
        bs=s[:,:,0].sum(axis=1)/s[:,:,1].sum(axis=1)
        # Each point averages within participants first; sample composition can vary by energy.
        curve=g.groupby(['group','laser_power']).rating.mean().reset_index().groupby('laser_power').rating.agg(['mean','count'])
        res[ds]={'estimable':True,'n':len(g),'subjects':len(a),'within_subject_slope_rating_per_joule':float(a[:,0].sum()/a[:,1].sum()),'ci95':np.quantile(bs,[.025,.975]).tolist(),'median_individual_slope':float(np.median(a[:,2])),'fraction_positive_individual_slopes':float((a[:,2]>0).mean()),'descriptive_curve':[{'joules':float(i),'mean_rating':float(row['mean']),'subjects':int(row['count'])} for i,row in curve.iterrows()]}
    return res

def run_dose(args):
    p=Path(args.out)/'results.json';r=read_json(p) if p.exists() else {}
    r['dose_response']=dose_analysis(load_zhao(args.data));write_json(p,r)
    print(json.dumps(r['dose_response'],indent=2))

def stress_case(td,target,cols,family,params,ds):
    g=td[td.dataset==ds];train=td[(td.dataset!=ds)&(td.split=='dev')];test=g[g.split=='test'];local=g[g.split=='dev']
    yp,w=fit_predict(family,params,cols,train,test,target)
    lp,w=fit_predict(family,params,cols,local,test,target)
    base=np.repeat(train[target].mean(),len(test))
    report={'source_dev_to_target_test':metrics(test,test[target],yp,1000),'source_mean':metrics(test,test[target],base),'target_study_dev_to_test':metrics(test,test[target],lp,1000)}
    b=test[['source_row','dataset','group']].copy();b['y']=test[target];b['transfer_prediction']=yp;b['local_prediction']=lp
    return ds,report,b

def run_stress(args):
    d=split_subjects(load_zhao(args.data));path=Path(args.out)/'results.json';r=read_json(path)
    allpred=[]
    for task in args.tasks:
        td,target,cols=task_data(d,task);tr=r['tasks'][task]
        # Study-specific comparisons use already-selected parameters: supplementary only.
        st=tr.setdefault('stress',{})
        for x in tr['finalists']:
            family=x['family'];p=x['params'];rows=st.setdefault(family,{})
            pending=[ds for ds in sorted(td.dataset.unique()) if ds not in rows]
            reports=Parallel(n_jobs=args.jobs)(delayed(stress_case)(td,target,cols,family,p,ds) for ds in pending)
            for ds,report,b in reports:
                rows[ds]=report;allpred.append(b)
                print('TRANSFER',task,family,ds,report['source_dev_to_target_test']['r2'],flush=True)
                b['task']=task;b['model']=family
                write_json(path,r)
        # New subject + withheld internal energy: interpolation stress in dense-dose study only.
        if task=='stimulus':
            st['unseen_energy']={};g=td[td.dataset=='ds005473'];levels=sorted(g.laser_power.unique());omitted=levels[2:-2:3]
            train=g[(g.split=='dev')&~g.laser_power.isin(omitted)];test=g[(g.split=='test')&g.laser_power.isin(omitted)]
            for x in tr['finalists']:
                yp,w=fit_predict(x['family'],x['params'],cols,train,test,target)
                st['unseen_energy'][x['family']]={'heldout_joules':omitted,'train_n':len(train),'metrics':metrics(test,test[target],yp,1000)}
        if task=='pain':
            # Does EEG add value when the actual applied energy is already known?
            test=td[td.split=='test'];train=td[td.split=='dev'];st['known_stimulus_ablation']={}
            for x in tr['finalists']:
                name=x['family'];st['known_stimulus_ablation'][name]={}
                for label,cc in [('stimulus_only',['laser_power']),('eeg_plus_stimulus',cols+['laser_power'])]:
                    yp,w=fit_predict(name,x['params'],cc,train,test,target)
                    st['known_stimulus_ablation'][name][label]=metrics(test,test[target],yp,2000)
        write_json(path,r)
    if allpred:
        pp=Path(args.out)/'stress_predictions.csv.gz'
        blocks=([pd.read_csv(pp)] if pp.exists() else [])+allpred
        pd.concat(blocks,ignore_index=True).drop_duplicates(['source_row','task','model'],keep='last').to_csv(pp,index=False,compression='gzip')

# Public source download: never substitutes synthetic data on failure.
GOZZI_FILES={'Trials.pkl':'62a614b3737cfdeffbd479227f8d5f48','Subjects.csv':'99291f8716affe2e6ae3ebe0d290ca8a','SubjectAreas.csv':'8a4d850d6213cb20554a49dd018ed751'}
def download_gozzi(args):
    import urllib.request
    dest=Path(args.dest);dest.mkdir(parents=True,exist_ok=True)
    for name,digest in GOZZI_FILES.items():
        path=dest/name
        if path.exists() and hashlib.md5(path.read_bytes()).hexdigest()==digest:print('Verified',name);continue
        url='https://zenodo.org/records/12570277/files/'+name+'?download=1'
        try:
            with urllib.request.urlopen(url,timeout=60) as req:body=req.read()
        except Exception as e:raise RuntimeError(f'Download blocked for {name}; download from https://zenodo.org/records/12570277 and place in {dest}. No external result has been generated.') from e
        if hashlib.md5(body).hexdigest()!=digest:raise ValueError(f'{name}: publisher MD5 mismatch; inspect source version')
        path.write_bytes(body);print('Verified',name)
    print('Data downloaded. Trials.pkl schema must be inspected before selecting EEG predictors/labels. Do not automatically unpickle untrusted files.')

def run_external(args):
    """Strict numeric CSV adapter; full schema supplied explicitly, no guessed labels."""
    d=pd.read_csv(args.csv);cols=args.features.split(',')
    required=[args.subject,args.target]+cols
    if not set(required)<=set(d):raise ValueError('Missing explicit subject/target/feature columns')
    if args.target in cols or args.subject in cols:raise ValueError('Target or subject leaked into predictors')
    if args.dataset_col and args.dataset_col in cols:raise ValueError('Dataset identifier cannot be an EEG predictor')
    if not all(pd.api.types.is_numeric_dtype(d[c]) for c in cols+[args.target]):raise ValueError('Features and continuous target must be numeric')
    if not d[args.target].dropna().between(args.target_min,args.target_max).all():raise ValueError('Out-of-range labels: audit them before training')
    if d[args.subject].isna().any():raise ValueError('Missing participant identity')
    d=d[d[args.target].notna()].copy();d['dataset']=d[args.dataset_col].astype(str) if args.dataset_col else args.name
    d['group']=d.dataset+'/'+d[args.subject].astype(str);d=split_subjects(d)
    dev=d[d.split=='dev'];test=d[d.split=='test'];cands=[]
    for family,pp in CANDIDATES.items():
        cands+=Parallel(n_jobs=args.jobs)(delayed(evaluate_candidate)(family,p,dev,args.target,cols) for p in pp)
    bests=sorted([min([c for c in cands if c['family']==f],key=lambda c:c['cv']['macro_mae']) for f in CANDIDATES],key=lambda c:c['cv']['macro_mae'])[:2]
    r={'source':args.source,'sha256':sha(args.csv),'target':args.target,'units':args.units,'features':cols,'candidates':cands,'test':{}}
    pred=test[[args.subject,'dataset','group',args.target]].copy()
    baseline=test.dataset.map(dev.groupby('dataset')[args.target].mean()).fillna(dev[args.target].mean()).to_numpy()
    r['test']['study_mean']=metrics(test,test[args.target],baseline,2000);pred['study_mean']=baseline
    for x in bests:
        p,w=fit_predict(x['family'],x['params'],cols,dev,test,args.target);r['test'][x['family']]=metrics(test,test[args.target],p,2000);pred[x['family']]=p
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);write_json(out/'results.json',r);pred.to_csv(out/'predictions.csv.gz',index=False,compression='gzip')

def run_fit(args):
    import joblib
    d=load_zhao(args.data);td,target,cols=task_data(d,args.task);r=read_json(Path(args.out)/'results.json')
    if args.dataset:
        td=td[td.dataset==args.dataset].copy()
        if td.empty:raise ValueError('Requested dataset is absent from this task')
    x=r['tasks'][args.task]['finalists'][args.rank-1];m=make_model(x['family'],x['params'],cols)
    with threadpool_limits(1):m.fit(td[cols],td[target])
    joblib.dump({'model':m,'features':cols,'target':target,'family':x['family'],'dataset':args.dataset,'fit_scope':'all available data; do not use for reported test evaluation','data_sha256':sha(args.data)},args.save,compress=3)
    print('Saved full-data model:',args.save)

def run_predict(args):
    import joblib
    b=joblib.load(args.model);d=pd.read_csv(args.csv)
    if b.get('dataset') and 'dataset' in d and not d.dataset.eq(b['dataset']).all():
        raise ValueError('Input study does not match the saved study-specific model')
    p=b['model'].predict(d[b['features']]);pd.DataFrame({'prediction':p}).to_csv(args.save,index=False)

def self_check():
    d=split_subjects(load_zhao(ROOT/'data/zhao_features.csv'))
    assert set(FEATURES).isdisjoint({'rating','laser_power','subject','epoch','sfreq','dataset','gamma_band_hz'})
    assert d.groupby('group').split.nunique().max()==1
    assert not d.rating.dropna().gt(10).any()
    # Held-out extremes cannot alter fitted clipping thresholds.
    t=RobustFeatures(['x']).fit([[1],[2],[3],[4]]);lo=t.lo_.copy();hi=t.hi_.copy();t.transform([[1e20]])
    assert np.array_equal(lo,t.lo_) and np.array_equal(hi,t.hi_)
    for f,ps in CANDIDATES.items():
        m=make_model(f,ps[0],FEATURES);m.fit(d[FEATURES].iloc[:80],d.laser_power.iloc[:80]);assert np.isfinite(m.predict(d[FEATURES].iloc[80:85])).all()
    print('Passed split isolation, feature leakage, label guard, train-only transform, all model fit/predict checks.')

def run_summarize(args):
    """Recompute descriptive dose curves, calibration analysis, and one figure."""
    root=Path(args.out);path=root/'results.json';r=read_json(path)
    d=load_zhao(args.data);r['dose_response']=dose_analysis(d)
    raw=pd.read_csv(args.data)
    bad=raw[FEATURES].duplicated(keep=False)
    r['audit']={'source_rows':len(raw),'retained_rows':len(d),'retained_subject_groups':int(d.group.nunique()),'missing_ratings':int(raw.rating.isna().sum()),'invalid_ratings':raw.loc[raw.rating.notna()&~raw.rating.between(0,10),['dataset','subject','epoch','rating']].to_dict('records'),'duplicate_feature_rows':int(bad.sum()),'quarantined_subjects':raw.loc[bad,['dataset','subject']].drop_duplicates().to_dict('records'),'excluded_rows':len(raw)-len(d),'note':'All supplied EEG-derived features remain unverified against original waveforms.'}
    pred=pd.read_csv(root/'predictions.csv.gz')
    for task,t in r['tasks'].items():
        pp=pred[pred.task==task];t['paired_vs_study_mean']={}
        for x in t['finalists']:
            name=x['family'];q=pp[['dataset','group']].copy();q['delta']=abs(pp.y-pp[name])-abs(pp.y-pp.study_mean)
            means=q.groupby(['dataset','group']).delta.mean();samples=np.zeros(2000);rng=np.random.default_rng(SEED)
            for ds in means.index.get_level_values(0).unique():
                v=means.loc[ds].to_numpy();samples+=rng.choice(v,(2000,len(v))).mean(axis=1)
            samples/=means.index.get_level_values(0).nunique()
            t['paired_vs_study_mean'][name]={'macro_mae_difference':float(means.groupby(level=0).mean().mean()),'ci95':np.quantile(samples,[.025,.975]).tolist()}
    if 'pain' in r['tasks']:
        pain=pred[pred.task=='pain'].sort_values(['group','epoch']).copy()
        rank=pain.groupby('group').cumcount();cnt=pain.groupby('group').group.transform('size')
        calib=pain[(rank<5)&(cnt>5)];ev=pain[(rank>=5)&(cnt>5)].copy()
        r['tasks']['pain']['five_trial_calibration']={'calibration_trials_per_person':5,'evaluation_rows':len(ev),'excluded_subjects_with_at_most_five_trials':int(pain.loc[cnt<=5,'group'].nunique()),'models':{}}
        cc=r['tasks']['pain']['five_trial_calibration']['models']
        base=ev.group.map(calib.groupby('group').y.mean()).to_numpy()
        cc['calibration_mean']=metrics(ev,ev.y,base,2000)
        for x in r['tasks']['pain']['finalists']:
            name=x['family'];offset=(calib.y-calib[name]).groupby(calib.group).mean()
            # Fixed mild shrinkage: 5 observations / (5 + 5 prior pseudo-observations).
            pp=ev[name].to_numpy()+.5*ev.group.map(offset).to_numpy()
            cc[name]={'without_calibration_same_rows':metrics(ev,ev.y,ev[name],2000),'with_offset':metrics(ev,ev.y,pp,2000)}
        r['tasks']['pain']['five_trial_calibration']['note']='Exploratory; earlier retained epoch order is only a chronology proxy. No calibration trial is scored. Offset uses only the first five observed ratings, never evaluation ratings.'
    r['delivered_code_sha256']=sha(__file__);write_json(path,r)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    colors=['#1565C0','#EF6C00','#2E7D32']
    for panel,task,unit in [(ax[0,0],'stimulus','J'),(ax[0,1],'pain','rating points')]:
        t=r['tasks'][task];names=['study_mean']+[x['family'] for x in t['finalists']]
        vals=[t['test'][n]['macro_mae'] for n in names];ci=np.array([t['test'][n]['macro_mae_ci95'] for n in names])
        labels=[n.replace('_',' ').title() for n in names]
        panel.barh(labels,vals,color=['#90A4AE']+colors[:2]);panel.errorbar(vals,range(3),xerr=np.maximum(0,np.array([np.array(vals)-ci[:,0],ci[:,1]-np.array(vals)])),fmt='none',ecolor='#263238',capsize=4)
        panel.invert_yaxis();panel.set_xlabel(f'Equal-study, equal-participant MAE ({unit}); lower is better')
        panel.set_title('EEG → '+('stimulus intensity' if task=='stimulus' else 'reported pain'),fontweight='bold')
        for j,v in enumerate(vals):panel.text(ci[j,1]+.012*max(ci[:,1]),j,f'{v:.3f}',va='center')
        panel.set_xlim(0,max(ci[:,1])*1.2)
    panel=ax[1,0]
    for i,ds in enumerate(['ds005473','ds005293']):
        q=r['dose_response'][ds]['descriptive_curve'];q=[v for v in q if v['subjects']>=3]
        panel.plot([v['joules'] for v in q],[v['mean_rating'] for v in q],'-o',color=colors[i],label=ds,ms=4)
    panel.set(xlabel='Applied laser energy (J)',ylabel='Observed mean pain rating (0–10)',ylim=(0,10.3));panel.set_title('Observed stimulus–pain relationship',fontweight='bold');panel.legend(frameon=False);panel.text(.02,.97,'Descriptive; ≥3 participants per energy\nEqual participant weighting; not a causal curve',transform=panel.transAxes,va='top',fontsize=9)
    panel=ax[1,1];t=r['tasks']['stimulus'];studies=list(t['by_study']['study_mean']);ypos=np.arange(len(studies));name=t['finalists'][0]['family']
    views=[('Pooled '+name.replace('_',' ').title(),[t['by_study'][name][ds]['r2'] for ds in studies]),('Study-specific '+name.replace('_',' ').title(),[t['stress'][name][ds]['target_study_dev_to_test']['r2'] for ds in studies])]
    for j,(label,vals) in enumerate(views):
        panel.barh(ypos+(j-.5)*.32,vals,height=.3,label=label,color=colors[j])
    panel.set_yticks(ypos,studies);panel.axvline(0,color='black',linewidth=.8);panel.set_xlabel('Held-out-participant R²');panel.set_title('Pooled vs study-specific stimulus models',fontweight='bold');panel.legend(frameon=False,fontsize=9,loc='upper left')
    fig.suptitle('Continuous EEG regression | actual held-out results',fontsize=17,fontweight='bold')
    fig.savefig(root/'results.png',dpi=170);plt.close(fig)
    print('Saved results.json and results.png')

# Independent Gozzi regression. Physical laser energy is NOT present in this release.
GOZZI_MD5={'Trials.pkl':'62a614b3737cfdeffbd479227f8d5f48','Subjects.csv':'99291f8716affe2e6ae3ebe0d290ca8a','SubjectAreas.csv':'8a4d850d6213cb20554a49dd018ed751'}

def prepare_gozzi(folder):
    folder=Path(folder)
    for name,digest in GOZZI_MD5.items():
        if hashlib.md5((folder/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'{name}: not the verified publisher release; inspect before loading pickle')
    raw=pd.read_pickle(folder/'Trials.pkl').reset_index(drop=True)
    s=pd.read_csv(folder/'Subjects.csv');areas=pd.read_csv(folder/'SubjectAreas.csv')
    key=['id','Area','B','iTrial'];phys=[c for c in raw if c.startswith(('EEG','SC'))]
    assert not raw.duplicated(key+['pain']).any()
    assert not s.id.duplicated().any() and not areas.duplicated(['id','Area']).any()
    raw['source_row']=np.arange(len(raw))
    ev=raw[raw.pain==1].copy();base=raw[raw.pain==0].copy()
    bad=sorted(ev.loc[ev[phys].duplicated(keep=False),'id'].unique())
    ev=ev[~ev.id.isin(bad)].copy()
    d=ev.merge(base[key+phys+['NRS']],on=key,suffixes=('','_base'),validate='one_to_one')
    assert (d.NRS==d.NRS_base).all() and d.NRS.between(0,10).all()
    eeg=[c for c in phys if c.startswith('EEG')]
    # Label-free per-trial transforms. No aggregation over held-out subjects.
    made={}
    for c in phys:
        scale=1e6 if c.startswith(('SCH','SCF')) and not any(k in c for k in ['Time','NPeaks','SampEn']) else 1.
        a=d[c].to_numpy()*scale;b=d[c+'_base'].to_numpy()*scale
        if c.startswith('EEG') and c not in ['EEG_mean','EEG_skew']:
            a=np.log(np.maximum(a,1e-12));b=np.log(np.maximum(b,1e-12))
        else:a=np.sign(a)*np.log1p(abs(a));b=np.sign(b)*np.log1p(abs(b))
        made['ev_'+c]=a;made['base_'+c]=b;made['change_'+c]=a-b
    bands=['delta','theta','alpha','beta','gamma']
    for prefix,suffix in [('ev',''),('base','_base')]:
        p=d[['EEG_'+b+suffix for b in bands]].to_numpy();p=p/p.sum(axis=1,keepdims=True)
        for j,b in enumerate(bands):made[prefix+'_relative_'+b]=p[:,j]
    features=pd.DataFrame(made,index=d.index)
    meta=['age','Gender','BMI','cohort_CRPS','cohort_HC','cohort_LBP','cohort_SCI_NP']
    d=d.merge(s[['id']+meta],on='id',validate='many_to_one')
    # Area file is audited and bundled; QST/CPM/TS outcomes are not predictors.
    assert d.merge(areas[['id','Area']],on=['id','Area'],validate='many_to_one').shape[0]==len(d)
    for a in [1,2,3]:features['area_'+str(a)]=(d.Area==a).astype(float)
    for c in meta:features['context_'+c]=d[c].to_numpy()
    out=pd.concat([d[['source_row','id','Area','B','iTrial','NRS']].reset_index(drop=True),features.reset_index(drop=True)],axis=1)
    out['subject']=out.id;out['group']='gozzi/'+out.id;out['dataset']='gozzi';out['epoch']=out.iTrial
    out=split_subjects(out)
    evcols=['ev_'+c for c in eeg]+['ev_relative_'+b for b in bands]
    paired=[c for c in features if ('EEG' in c or '_relative_' in c)]
    multimodal=[c for c in features if c.startswith(('ev_','base_','change_'))]
    context=[c for c in features if c.startswith(('context_','area_'))]
    views={'eeg':evcols,'paired_eeg':paired,'multimodal':multimodal,'multimodal_context':multimodal+context}
    audit={'raw_rows':len(raw),'raw_subjects':int(raw.id.nunique()),'event_rows_before_quarantine':len(ev)+int((raw.pain.eq(1)&raw.id.isin(bad)).sum()),'quarantined_subjects':bad,'retained_trials':len(out),'retained_subjects':int(out.group.nunique()),'raw_pairs_share_NRS':True,'no_physical_intensity_column':True,'md5':GOZZI_MD5,'feature_views':views,'source':'https://zenodo.org/records/12570277','semantics':'pain=1 event rows; pain=0 paired baseline rows, not zero-rated pain. Physical intensity unavailable. Source release has more areas/trials than publication summary.'}
    return out,views,audit

def gozzi_grid():
    return {
      'ridge':[{'alpha':a} for a in [1,100,10000]],
      'spline_ridge':[{'n_knots':3,'alpha':a} for a in [100,1000]],
      'svr':[{'C':c,'gamma':g} for c,g in [(1,.01),(10,.01),(1,.05)]],
      'extra_trees':[{'min_samples_leaf':l,'max_features':m} for l,m in [(3,.8),(15,1.),(50,1.)]],
      'random_forest':[{'min_samples_leaf':10,'max_features':.8}],
      'hist_gradient_boosting':[{'max_leaf_nodes':l,'l2_regularization':r,'max_iter':n} for l,r,n in [(7,10,200),(15,10,300),(15,100,500),(31,100,300)]],
      'mlp':[{'hidden_layer_sizes':h,'alpha':a} for h,a in [((64,32),10.),((32,),10.)]]}

def gozzi_candidate(family,params,dev,cols):
    pred=np.zeros(len(dev));conv=False;start=time.time()
    for k in range(3):
        b=dev.fold==k;p,w=fit_predict(family,params,cols,dev[~b],dev[b],'NRS');pred[np.where(b)[0]]=np.clip(p,0,10);conv|=w
    r={'family':family,'params':params,'cv':metrics(dev,dev.NRS,pred),'convergence_warning':conv,'seconds':round(time.time()-start,2)}
    print('GOZZI CV',family,params,'R2',round(r['cv']['r2'],4),flush=True)
    return r,pred

def r2_cluster_ci(d,y,p):
    z=pd.DataFrame({'g':np.asarray(d.group),'y':np.asarray(y),'p':np.asarray(p)})
    z['sse']=(z.y-z.p)**2;z['y2']=z.y**2
    q=z.groupby('g').agg(n=('y','size'),sy=('y','sum'),sy2=('y2','sum'),sse=('sse','sum')).to_numpy()
    rng=np.random.default_rng(SEED);q=q[rng.integers(0,len(q),(2000,len(q)))].sum(axis=1)
    v=1-q[:,3]/(q[:,2]-q[:,1]**2/q[:,0])
    return np.quantile(v,[.025,.975]).tolist()

def calibrate_predictions(d,p,mode,strength):
    """10 earliest retained trials in B=1 per area; evaluate B=2 only.
    Block order is a proxy; no timestamps are provided. Never uses evaluation NRS.
    """
    d=d.copy();d['p']=p
    cal=d[d.B==1].sort_values(['group','Area','iTrial']).groupby(['group','Area']).head(10)
    ev=d[d.B==2].copy();pred=[];inds=[]
    for (g,a),q in ev.groupby(['group','Area'],sort=False):
        c=cal[(cal.group==g)&(cal.Area==a)]
        if len(c)<5:continue
        if mode=='mean':v=np.repeat(c.NRS.mean(),len(q))
        elif mode=='offset':v=q.p.to_numpy()+(c.NRS-c.p).sum()/(len(c)+strength)
        elif mode=='affine':
            xx=np.c_[np.ones(len(c)),c.p.to_numpy()-5];zz=np.c_[np.ones(len(q)),q.p.to_numpy()-5]
            coef=np.linalg.solve(xx.T@xx+np.diag([strength,strength]),xx.T@(c.NRS-c.p))
            v=q.p.to_numpy()+zz@coef
        pred.extend(np.clip(v,0,10));inds.extend(q.index)
    return d.loc[inds].copy(),np.array(pred)

def run_gozzi(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    d,views,audit=prepare_gozzi(args.folder);dev=d[d.split=='dev'];test=d[d.split=='test']
    path=out/'gozzi_results.json';cache=out/'gozzi_cv_cache.joblib'
    import joblib
    r={'audit':audit,'seed':SEED,'selection':'minimum pooled development OOF MSE (maximum R2), 3 participant folds, bounded 0-10 outputs','train_n':len(dev),'test_n':len(test),'train_subjects':int(dev.group.nunique()),'test_subjects':int(test.group.nunique()),'views':{},'code_sha256':sha(__file__)}
    cached=joblib.load(cache) if cache.exists() else {}
    predictions=[];global_rank=[]
    for view,cols in views.items():
        if view not in cached:
            vals=Parallel(n_jobs=args.jobs)(delayed(gozzi_candidate)(f,p,dev,cols) for f,ps in gozzi_grid().items() for p in ps)
            cached[view]=vals;joblib.dump(cached,cache)
        vals=cached[view]
        bests=sorted([min([v for v in vals if v[0]['family']==f],key=lambda x:x[0]['cv']['rmse']) for f in gozzi_grid()],key=lambda x:x[0]['cv']['rmse'])
        r['views'][view]={'features':cols,'candidates':[v[0] for v in vals],'family_ranking':[v[0] for v in bests]}
        for candidate,oof in bests[:2]:global_rank.append((candidate['cv']['rmse'],view,candidate,oof))
    # Freeze ALL choices and calibration settings using development predictions only.
    global_rank.sort(key=lambda t:t[0]);chosen=[]
    for x in global_rank:
        if x[2]['family'] not in [q[2]['family'] for q in chosen]:chosen.append(x)
        if len(chosen)==2:break
    r['overall_finalists']=[{'view':v,'candidate':c} for _,v,c,_ in chosen]
    calibration=[]
    for _,view,c,oof in chosen:
        options=[]
        for mode in ['offset','affine']:
            for strength in [1.,5.,20.,100.]:
                ev,p=calibrate_predictions(dev,oof,mode,strength)
                options.append({'mode':mode,'strength':strength,'cv':metrics(ev,ev.NRS,p)})
        calibration.append(min(options,key=lambda q:q['cv']['rmse']))
    r['calibration_selection']=calibration;write_json(path,r)
    baseline=np.repeat(dev.NRS.mean(),len(test));r['mean_baseline']=metrics(test,test.NRS,baseline)
    def record(view,name,q,p):
        v=q[['source_row','group','subject','Area','B','iTrial','NRS']].copy();v['view']=view;v['model']=name;v['prediction']=p;predictions.append(v)
        m=metrics(q,q.NRS,p,2000);m['r2_ci95']=r2_cluster_ci(q,q.NRS,p);return m
    record('baseline','training_mean',test,baseline)
    fitted={}
    for view,vr in r['views'].items():
        vr['test']={}
        for c in vr['family_ranking'][:2]:
            m=make_model(c['family'],c['params'],views[view]);m.fit(dev[views[view]],dev.NRS);p=np.clip(m.predict(test[views[view]]),0,10)
            fitted[(view,c['family'])]=(m,p)
            vr['test'][c['family']]=record(view,c['family'],test,p)
            print('GOZZI TEST',view,c['family'],vr['test'][c['family']],flush=True)
    r['personalized']={}
    for (_,view,c,_),cfg in zip(chosen,calibration):
        m,p=fitted[(view,c['family'])];ev,cp=calibrate_predictions(test,p,cfg['mode'],cfg['strength'])
        name=view+'/'+c['family'];r['personalized'][name]={'configuration':cfg,'calibrated':record(view,c['family']+'_calibrated',ev,cp),'uncalibrated_same_rows':record(view,c['family']+'_uncalibrated_B2',ev,ev.p.to_numpy())}
        ev,base=calibrate_predictions(test,p,'mean',0);r['personalized'][name]['calibration_mean_baseline']=record(view,c['family']+'_calibration_mean',ev,base)
    # Export development-only fitted finalists, preserving test integrity.
    bundle={'kind':'gozzi_pain','target':'NRS','bounds':[0,10],'audit':audit,'models':[]}
    for (_,view,c,_),cfg in zip(chosen,calibration):
        m,_=fitted[(view,c['family'])];bundle['models'].append({'view':view,'family':c['family'],'features':views[view],'model':m,'calibration':cfg})
    joblib.dump(bundle,out/'gozzi_models.joblib',compress=3)
    pd.concat(predictions,ignore_index=True).to_csv(out/'gozzi_predictions.csv.gz',index=False)
    r['completed']=True;write_json(path,r)
    print('GOZZI COMPLETE',flush=True)

def run_gozzi_predict(args):
    import joblib
    b=joblib.load(args.model);d=pd.read_csv(args.csv)
    for i,m in enumerate(b['models']):
        missing=set(m['features'])-set(d)
        if missing:raise ValueError('Input must be prepare-gozzi feature CSV; missing '+str(sorted(missing)))
        d['prediction_rank_'+str(i+1)]=np.clip(m['model'].predict(d[m['features']]),0,10)
    d.to_csv(args.save,index=False)

def run_prepare_gozzi(args):
    d,views,audit=prepare_gozzi(args.folder);d.to_csv(args.save,index=False);print(json.dumps(audit,indent=2))

def run_stimulus_refine(args):
    """Follow-up study-specific search; original Zhao test set was already observed."""
    d=split_subjects(load_zhao(args.data));out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    results={};blocks=[]
    import joblib
    cache=out/'stimulus_refine_cache.joblib';cached=joblib.load(cache) if cache.exists() else {}
    for ds in ['ds005293','ds005473']:
        q=d[d.dataset==ds].copy();q['NRS']=q.laser_power;dev=q[q.split=='dev'];test=q[q.split=='test']
        if ds not in cached:
            # Intensities here lie inside 0..10, so the Gozzi CV helper's clipping is inactive.
            cached[ds]=Parallel(n_jobs=args.jobs)(delayed(gozzi_candidate)(f,p,dev,FEATURES) for f,ps in gozzi_grid().items() for p in ps)
            joblib.dump(cached,cache)
        candidates=cached[ds];ranks=sorted([min([v for v in candidates if v[0]['family']==f],key=lambda x:x[0]['cv']['rmse']) for f in gozzi_grid()],key=lambda x:x[0]['cv']['rmse'])
        results[ds]={'candidates':[c for c,_ in candidates],'family_ranking':[c for c,_ in ranks],'test':{}}
        for c,_ in ranks[:2]:
            p,_=fit_predict(c['family'],c['params'],FEATURES,dev,test,'laser_power');m=metrics(test,test.laser_power,p,2000);m['r2_ci95']=r2_cluster_ci(test,test.laser_power,p)
            results[ds]['test'][c['family']]=m
            b=test[['source_row','dataset','subject','group','epoch','laser_power']].copy();b['model']=c['family'];b['prediction']=p;blocks.append(b)
            print('REFINED STIMULUS TEST',ds,c['family'],m,flush=True)
    write_json(out/'stimulus_refined.json',{'selection':'development OOF RMSE','limitation':'Follow-up analysis on the previously viewed Zhao test participants, not fresh confirmatory validation. No selection uses test scores.','studies':results})
    pd.concat(blocks,ignore_index=True).to_csv(out/'stimulus_refined_predictions.csv.gz',index=False)

def common_bands(d,source):
    bands=['delta','theta','alpha','beta','gamma'];cols=['shared_'+b for b in bands]
    if source=='zhao':p=d[['psd_'+b for b in bands]].to_numpy()
    else:
        # Inverse of prepare_gozzi's deterministic log band transform.
        p=np.exp(d[['ev_EEG_'+b for b in bands]].to_numpy())
    p=np.maximum(p,1e-12);lp=np.log(p);clr=lp-lp.mean(axis=1,keepdims=True)
    d=d.copy();d[cols]=clr
    return d,cols

def transfer_fit(family,params,cols,target,source,test,joint):
    m=make_model(family,params,cols)
    if joint:
        train=pd.concat([target,source],ignore_index=True)
        weight=np.r_[np.ones(len(target)),np.repeat(len(target)/len(source),len(source))]
        last=m.regressor.steps[-1][0]
        m.fit(train[cols],train.NRS,**{last+'__sample_weight':weight})
    else:m.fit(target[cols],target.NRS)
    return np.clip(m.predict(test[cols]),0,10)

def transfer_candidate(family,params,cols,dev,source,joint):
    p=np.zeros(len(dev))
    for k in range(3):
        mask=dev.fold==k;p[np.where(mask)[0]]=transfer_fit(family,params,cols,dev[~mask],source,dev[mask],joint)
    r={'family':family,'params':params,'cv':metrics(dev,dev.NRS,p)}
    print('TRANSFER CV',joint,family,params,r['cv']['r2'],flush=True);return r

def run_joint(args):
    z=split_subjects(load_zhao(args.data));z=z[z.rating.notna()].copy();z['NRS']=z.rating
    g,_,_=prepare_gozzi(args.folder);z,cols=common_bands(z,'zhao');g,_=common_bands(g,'gozzi')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);results={};blocks=[]
    grid={'ridge':[{'alpha':a} for a in [1,100,10000]],'extra_trees':[{'min_samples_leaf':50,'max_features':1.}]}
    for name,t,s in [('gozzi_plus_zhao',g,z),('zhao_plus_gozzi',z,g)]:
        dev=t[t.split=='dev'];test=t[t.split=='test'];source=s[s.split=='dev']
        results[name]={'features':cols,'target_dev_n':len(dev),'source_dev_n':len(source),'test':{},'candidates':{}}
        for joint in [False,True]:
            label='joint' if joint else 'target_only'
            candidates=Parallel(n_jobs=args.jobs)(delayed(transfer_candidate)(f,p,cols,dev,source,joint) for f,ps in grid.items() for p in ps)
            c=min(candidates,key=lambda q:q['cv']['rmse']);results[name]['candidates'][label]=candidates
            p=transfer_fit(c['family'],c['params'],cols,dev,source,test,joint)
            m=metrics(test,test.NRS,p,2000);m['r2_ci95']=r2_cluster_ci(test,test.NRS,p)
            results[name]['test'][label]={'selected':c,'metrics':m}
            b=test[['source_row','group','NRS']].copy();b['direction']=name;b['model']=label;b['prediction']=p;blocks.append(b)
            print('TRANSFER TEST',name,label,m,flush=True)
    write_json(out/'joint_results.json',{'note':'Exploratory approximate centered-log-ratio five-band alignment; exact band definitions, channel and extraction windows are not harmonized. Both datasets contribute only development participants. Each source has equal total training weight. No physical stimulus labels added.','directions':results})
    pd.concat(blocks,ignore_index=True).to_csv(out/'joint_predictions.csv.gz',index=False)

def plot_updated_results(out,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    blue='#1769AA';orange='#E67E22';gray='#8A9AA4'
    a=ax[0,0];ds=['ds005293','ds005473'];x=np.arange(2)
    for k,(family,color) in enumerate([('extra_trees',blue),('random_forest',orange)]):
        ms=[r['stimulus_refined']['studies'][d]['test'][family] for d in ds];v=np.array([m['r2'] for m in ms]);ci=np.array([m['r2_ci95'] for m in ms])
        a.bar(x+(k-.5)*.32,v,width=.3,label=family.replace('_',' ').title(),color=color)
        a.errorbar(x+(k-.5)*.32,v,yerr=np.maximum(0,np.array([v-ci[:,0],ci[:,1]-v])),fmt='none',ecolor='#263238',capsize=4)
    a.axhline(.75,color='#A93226',ls='--',label='Requested R² = 0.75');a.set_xticks(x,ds);a.set(ylabel='Held-out R²',ylim=(0,.85));a.legend(frameon=True,facecolor='white',framealpha=1,fontsize=9);a.set_title('Physical intensity: follow-up Zhao results',fontweight='bold')
    a=ax[0,1];labels=['EEG','EEG + paired baseline','EEG + skin conductance','Multimodal + context'];vals=[];cis=[]
    for view in ['eeg','paired_eeg','multimodal','multimodal_context']:
        q=r['gozzi']['views'][view];f=q['family_ranking'][0]['family'];m=q['test'][f];vals.append(m['r2']);cis.append(m['r2_ci95'])
    v=np.array(vals);ci=np.array(cis);a.barh(labels,v,color=blue);a.errorbar(v,range(4),xerr=np.maximum(0,np.array([v-ci[:,0],ci[:,1]-v])),fmt='none',ecolor='#263238',capsize=4);a.axvline(0,color=gray);a.invert_yaxis();a.set_xlabel('Held-out R²; 24 test participants');a.set_title('Independent Gozzi pain prediction',fontweight='bold')
    a=ax[1,0]
    initial=next(iter(r['gozzi']['personalized'].values()))
    labels=['Calibration mean','Spline + Ridge + offset'];ms=[initial['calibration_mean_baseline'],initial['calibrated']]
    if 'personalized' in r:
        for c in r['personalized']['family_ranking'][:2]:
            labels.append(c['family'].replace('_',' ').title()+' residual model');ms.append(r['personalized']['test'][c['family']])
    vals=np.array([m['r2'] for m in ms]);ci=np.array([m['r2_ci95'] for m in ms])
    a.barh(labels,vals,color=[gray,blue,orange,blue][:len(vals)]);a.errorbar(vals,range(len(vals)),xerr=np.maximum(0,np.array([vals-ci[:,0],ci[:,1]-vals])),fmt='none',ecolor='#263238',capsize=4)
    a.invert_yaxis();a.axvline(.75,color='#A93226',ls='--',linewidth=1);a.set(xlabel='R² on B=2 after per-area calibration',xlim=(0,.95));a.set_title('Personalized pain: exploratory follow-up',fontweight='bold')
    a=ax[1,1]
    for ds,color in [('ds005473',blue),('ds005293',orange)]:
        if 'dose_response' in r:
            q=[v for v in r['dose_response'][ds]['descriptive_curve'] if v['subjects']>=3];a.plot([v['joules'] for v in q],[v['mean_rating'] for v in q],'-o',ms=4,label=ds,color=color)
    a.set(xlabel='Applied laser energy (J)',ylabel='Observed mean pain (0–10)',ylim=(0,10.5));a.legend(frameon=False);a.set_title('Observed stimulus–pain relationship',fontweight='bold')
    fig.suptitle('Continuous regression: stimulus and pain are separate targets',fontsize=17,fontweight='bold')
    fig.supxlabel('R² panels use held-out participants; bars do not imply causal effects. Error bars: participant-bootstrap 95% intervals.',fontsize=10)
    fig.savefig(Path(out)/'results.png',dpi=170);plt.close(fig)

def run_consolidate(args):
    out=Path(args.out);r=read_json(out/'results.json') if (out/'results.json').exists() else {}
    for key,name in [('gozzi','gozzi_results.json'),('joint','joint_results.json'),('stimulus_refined','stimulus_refined.json'),('dose_predict','dose_predict_results.json'),('personalized','personalized_results.json')]:
        r[key]=read_json(out/name)
    r['delivered_code_sha256']=sha(__file__);write_json(out/'results.json',r)
    blocks=[]
    for exp,name in [('gozzi','gozzi_predictions.csv.gz'),('joint','joint_predictions.csv.gz'),('stimulus_refined','stimulus_refined_predictions.csv.gz'),('dose_predict','dose_predict_predictions.csv.gz'),('personalized','personalized_predictions.csv.gz')]:
        q=pd.read_csv(out/name);q['experiment']=exp;blocks.append(q)
    pd.concat(blocks,ignore_index=True).to_csv(out/'external_predictions.csv.gz',index=False)
    plot_updated_results(out,r)
    print('Consolidated results.json, external_predictions.csv.gz and results.png')

def run_fit_refined(args):
    import joblib
    d=load_zhao(args.data);d=d[d.dataset==args.dataset].copy()
    r=read_json(Path(args.out)/'results.json')['stimulus_refined']['studies'][args.dataset]
    c=r['family_ranking'][args.rank-1];m=make_model(c['family'],c['params'],FEATURES)
    with threadpool_limits(1):m.fit(d[FEATURES],d.laser_power)
    joblib.dump({'model':m,'features':FEATURES,'target':'laser_power','family':c['family'],'dataset':args.dataset,'fit_scope':'all available data; not a test-evaluation model','data_sha256':sha(args.data)},args.save,compress=3)
    print('Saved',args.save)

def run_dose_predict(args):
    d=split_subjects(load_zhao(args.data));d=d[d.rating.notna()].copy();d['NRS']=d.rating
    # Freeze this supplementary protocol before its evaluation: energy only, six configurations,
    # two studies, 10 earlier retained epochs/person for optional calibration.
    grid={'ridge':[{'alpha':a} for a in [.01,1.,100.]],'spline_ridge':[{'n_knots':k,'alpha':1.} for k in [3,5]],'hist_gradient_boosting':[{'max_leaf_nodes':7,'l2_regularization':10,'max_iter':200,'monotonic_cst':[1]}]}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);res={};blocks=[]
    for ds in ['ds005293','ds005473']:
        q=d[d.dataset==ds].sort_values(['group','epoch']).copy();q['Area']=1;q['B']=np.where(q.groupby('group').cumcount()<10,1,2);q['iTrial']=q.epoch
        dev=q[q.split=='dev'];test=q[q.split=='test'];cols=['laser_power']
        cs=Parallel(n_jobs=args.jobs)(delayed(gozzi_candidate)(f,p,dev,cols) for f,ps in grid.items() for p in ps)
        ranks=sorted([min([v for v in cs if v[0]['family']==f],key=lambda v:v[0]['cv']['rmse']) for f in grid],key=lambda v:v[0]['cv']['rmse'])
        res[ds]={'predictors':cols,'target':'rating','candidates':[c for c,_ in cs],'test':{}}
        for c,oof in ranks[:2]:
            settings=[]
            for mode in ['offset','affine']:
                for strength in [1.,5.,20.,100.]:
                    ev,p=calibrate_predictions(dev,oof,mode,strength);settings.append({'mode':mode,'strength':strength,'cv':metrics(ev,ev.NRS,p)})
            cfg=min(settings,key=lambda v:v['cv']['rmse']);p,_=fit_predict(c['family'],c['params'],cols,dev,test,'NRS');p=np.clip(p,0,10)
            ev,cal=calibrate_predictions(test,p,cfg['mode'],cfg['strength']);_,base=calibrate_predictions(test,p,'mean',0)
            answer={'selected':c,'calibration':cfg}
            for label,t,pred in [('uncalibrated_all',test,p),('uncalibrated_later',ev,ev.p.to_numpy()),('calibrated_later',ev,cal),('calibration_mean_later',ev,base)]:
                m=metrics(t,t.NRS,pred,2000);m['r2_ci95']=r2_cluster_ci(t,t.NRS,pred);answer[label]=m
                b=t[['source_row','dataset','group','subject','epoch','laser_power','NRS']].copy();b['model']=c['family']+'/'+label;b['prediction']=pred;blocks.append(b)
            res[ds]['test'][c['family']]=answer
            print('DOSE PREDICT TEST',ds,c['family'],answer,flush=True)
    write_json(out/'dose_predict_results.json',{'note':'Known laser energy -> reported pain, not EEG -> stimulus. Exploratory follow-up on previously viewed Zhao participants; chronological order is only retained epoch order. Calibration excludes first 10 retained trials/person from scoring. No Gozzi physical-intensity labels exist.','studies':res})
    pd.concat(blocks,ignore_index=True).to_csv(out/'dose_predict_predictions.csv.gz',index=False)

def personal_features(d,cols):
    """Use ONLY first 10 B=1 trials per subject/area as labeled calibration.
    B=2 labels are copied for scoring/training, never used to make predictors.
    """
    d=d.copy();cal=d[d.B==1].sort_values(['group','Area','iTrial']).groupby(['group','Area']).head(10)
    ev=d[d.B==2].copy();key=['group','Area'];stats=cal.groupby(key).NRS.agg(['mean','std','count'])
    means=cal.groupby(key)[cols].mean().add_prefix('cal_')
    ev=ev.merge(stats.rename(columns={'mean':'cal_rating_mean','std':'cal_rating_std','count':'cal_n'}),on=key,validate='many_to_one')
    ev=ev[ev.cal_n>=5].copy();ev=ev.merge(means,on=key,validate='many_to_one')
    additions={}
    for c in cols:additions['centered_'+c]=ev[c]-ev['cal_'+c]
    ev=pd.concat([ev,pd.DataFrame(additions,index=ev.index)],axis=1)
    features=cols+list(additions)+['cal_rating_mean','cal_rating_std']
    ev['residual_target']=ev.NRS-ev.cal_rating_mean
    return ev,features

def personal_candidate(family,params,dev,cols):
    pred=np.zeros(len(dev));converged=False
    for k in range(3):
        mask=dev.fold==k;p,w=fit_predict(family,params,cols,dev[~mask],dev[mask],'residual_target')
        pred[np.where(mask)[0]]=np.clip(p+dev.loc[mask,'cal_rating_mean'].to_numpy(),0,10);converged|=w
    c={'family':family,'params':params,'cv':metrics(dev,dev.NRS,pred),'convergence_warning':converged}
    print('PERSONAL CV',family,params,c['cv']['r2'],flush=True);return c

def run_personalized(args):
    import joblib
    d,views,audit=prepare_gozzi(args.folder);q,cols=personal_features(d,views['multimodal']);dev=q[q.split=='dev'];test=q[q.split=='test']
    grid={'ridge':[{'alpha':a} for a in [100.,1000.,10000.]],'spline_ridge':[{'n_knots':3,'alpha':1000.}], 'hist_gradient_boosting':[{'max_leaf_nodes':l,'l2_regularization':100,'max_iter':200} for l in [7,15]]}
    cs=Parallel(n_jobs=args.jobs)(delayed(personal_candidate)(f,p,dev,cols) for f,ps in grid.items() for p in ps)
    ranks=sorted([min([c for c in cs if c['family']==f],key=lambda c:c['cv']['rmse']) for f in grid],key=lambda c:c['cv']['rmse'])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);r={'note':'Exploratory follow-up on the already-viewed Gozzi test participants. Trains B=2 residuals against first-10-B=1 calibration mean. Features use event/baseline physiology and differences from B=1 calibration feature means, plus calibration rating mean/std. All model selection is development-subject CV. Not EEG-only, not physical-intensity prediction.','features':cols,'candidates':cs,'family_ranking':ranks,'test':{}}
    write_json(out/'personalized_results.json',r);blocks=[];bundle=joblib.load(out/'gozzi_models.joblib');bundle['personalized_models']=[]
    for c in ranks[:2]:
        m=make_model(c['family'],c['params'],cols);m.fit(dev[cols],dev.residual_target);pred=np.clip(m.predict(test[cols])+test.cal_rating_mean,0,10)
        met=metrics(test,test.NRS,pred,2000);met['r2_ci95']=r2_cluster_ci(test,test.NRS,pred);r['test'][c['family']]=met
        b=test[['source_row','group','subject','Area','B','iTrial','NRS']].copy();b['model']=c['family'];b['prediction']=pred;blocks.append(b)
        bundle['personalized_models'].append({'family':c['family'],'features':cols,'base_features':views['multimodal'],'model':m})
        print('PERSONAL TEST',c['family'],met,flush=True)
    write_json(out/'personalized_results.json',r);pd.concat(blocks,ignore_index=True).to_csv(out/'personalized_predictions.csv.gz',index=False);joblib.dump(bundle,out/'gozzi_models.joblib',compress=3)

def run_predict_personalized(args):
    import joblib
    bundle=joblib.load(args.model);d=pd.read_csv(args.csv)
    # NRS is required for the earlier calibration rows only. B=2 may be unlabeled.
    if 'NRS' not in d:raise ValueError('Supply NRS ratings on B=1 calibration rows; B=2 ratings can be missing')
    q,cols=personal_features(d,bundle['personalized_models'][0]['base_features'])
    out=q[['source_row','group','Area','B','iTrial']].copy()
    for i,m in enumerate(bundle['personalized_models']):out['prediction_rank_'+str(i+1)]=np.clip(m['model'].predict(q[m['features']])+q.cal_rating_mean,0,10)
    out.to_csv(args.save,index=False)

def main():
    a=argparse.ArgumentParser(description=__doc__);sp=a.add_subparsers(dest='command',required=True)
    for cmd in ['benchmark','stress','dose','fit','summarize']:
        p=sp.add_parser(cmd);p.add_argument('--data',type=Path,default=ROOT/'data/zhao_features.csv');p.add_argument('--out',type=Path,default=ROOT)
        if cmd in ('benchmark','stress'):p.add_argument('--tasks',nargs='+',choices=['stimulus','pain'],default=['stimulus','pain']);p.add_argument('--jobs',type=int,default=3)
        if cmd=='fit':p.add_argument('--task',choices=['stimulus','pain'],required=True);p.add_argument('--rank',type=int,choices=[1,2],default=1);p.add_argument('--save',required=True);p.add_argument('--dataset',default=None)
    p=sp.add_parser('download-gozzi');p.add_argument('--dest',default=ROOT/'data/gozzi')
    p=sp.add_parser('external');p.add_argument('--csv',required=True);p.add_argument('--subject',required=True);p.add_argument('--target',required=True);p.add_argument('--features',required=True);p.add_argument('--name',required=True);p.add_argument('--source',required=True);p.add_argument('--units',required=True);p.add_argument('--target-min',type=float,required=True);p.add_argument('--target-max',type=float,required=True);p.add_argument('--dataset-col');p.add_argument('--out',required=True);p.add_argument('--jobs',type=int,default=3)
    p=sp.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--csv',required=True);p.add_argument('--save',required=True)
    p=sp.add_parser('gozzi');p.add_argument('--folder',type=Path,default=ROOT/'data/gozzi');p.add_argument('--out',type=Path,default=ROOT);p.add_argument('--jobs',type=int,default=3)
    p=sp.add_parser('prepare-gozzi');p.add_argument('--folder',type=Path,default=ROOT/'data/gozzi');p.add_argument('--save',required=True)
    p=sp.add_parser('predict-gozzi');p.add_argument('--model',required=True);p.add_argument('--csv',required=True);p.add_argument('--save',required=True)
    for cmd in ['stimulus-refine','joint','dose-predict','personalized']:
        p=sp.add_parser(cmd);p.add_argument('--data',type=Path,default=ROOT/'data/zhao_features.csv');p.add_argument('--folder',type=Path,default=ROOT/'data/gozzi');p.add_argument('--out',type=Path,default=ROOT);p.add_argument('--jobs',type=int,default=2)
    p=sp.add_parser('consolidate');p.add_argument('--out',type=Path,default=ROOT)
    p=sp.add_parser('fit-refined');p.add_argument('--data',type=Path,default=ROOT/'data/zhao_features.csv');p.add_argument('--out',type=Path,default=ROOT);p.add_argument('--dataset',required=True,choices=['ds005293','ds005473']);p.add_argument('--rank',type=int,default=1,choices=[1,2]);p.add_argument('--save',required=True)
    p=sp.add_parser('predict-personalized');p.add_argument('--model',required=True);p.add_argument('--csv',required=True);p.add_argument('--save',required=True)
    sp.add_parser('self-check');args=a.parse_args()
    if args.command=='self-check':self_check()
    else:{'personalized':run_personalized,'predict-personalized':run_predict_personalized,'dose-predict':run_dose_predict,'consolidate':run_consolidate,'fit-refined':run_fit_refined,'stimulus-refine':run_stimulus_refine,'joint':run_joint,'gozzi':run_gozzi,'prepare-gozzi':run_prepare_gozzi,'predict-gozzi':run_gozzi_predict,'benchmark':run_benchmark,'stress':run_stress,'dose':run_dose,'download-gozzi':download_gozzi,'external':run_external,'fit':run_fit,'predict':run_predict,'summarize':run_summarize}[args.command](args)
if __name__=='__main__':main()
