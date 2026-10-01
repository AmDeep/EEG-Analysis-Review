#!/usr/bin/env python3
"""Raw EEG -> laser energy (J). Real data required for benchmark; smoke tests are synthetic."""
import argparse, hashlib, json, os, platform, sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
SEED=20260930

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,allow_nan=False))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rank_id(x):return hashlib.sha256(f'{SEED}/{x}'.encode()).hexdigest()
def get_json(url):
    with urllib.request.urlopen(url,timeout=60) as r:return json.load(r)
def download_file(url,dest,md5=None):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and md5 and hashlib.md5(dest.read_bytes()).hexdigest()==md5:return
    tmp=dest.with_suffix(dest.suffix+'.part')
    with urllib.request.urlopen(url,timeout=120) as src,open(tmp,'wb') as out:
        while True:
            b=src.read(1024*1024)
            if not b:break
            out.write(b)
    if md5 and hashlib.md5(tmp.read_bytes()).hexdigest()!=md5:raise ValueError('MD5 mismatch: '+str(dest))
    tmp.replace(dest);print('Downloaded',dest,flush=True)
def safe_path(root,rel):
    p=(Path(root)/rel).resolve()
    if not p.is_relative_to(Path(root).resolve()):raise ValueError('Unsafe remote path')
    return p

def download(args):
    # Public API only. No authentication bypass, no private download URLs guessed.
    dest=Path(args.dest)
    if args.source=='tiemann':
        def crawl(url,prefix=''):
            while url:
                page=get_json(url)
                for item in page['data']:
                    a=item['attributes'];rel=prefix+a['name']
                    if a['kind']=='folder':crawl(item['relationships']['files']['links']['related']['href'],rel+'/')
                    else:
                        md5=a.get('extra',{}).get('hashes',{}).get('md5')
                        download_file(item['links']['download'],safe_path(dest,rel),md5)
                url=page.get('links',{}).get('next')
        page=get_json('https://api.osf.io/v2/nodes/z2h86/files/')
        for provider in page['data']:
            if provider['attributes']['name']=='osfstorage':crawl(provider['relationships']['files']['links']['related']['href'])
    else:
        accession=args.source;token=None;ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
        while True:
            query={'list-type':'2','prefix':accession+'/'}
            if token:query['continuation-token']=token
            url='https://openneuro.org.s3.amazonaws.com/?'+urllib.parse.urlencode(query)
            with urllib.request.urlopen(url,timeout=60) as r:tree=ET.fromstring(r.read())
            for node in tree.findall('s:Contents',ns):
                key=node.find('s:Key',ns).text;rel=key[len(accession)+1:]
                if not rel or rel.endswith('/') or rel.startswith(('derivatives/','.git/')):continue
                download_file('https://openneuro.org.s3.amazonaws.com/'+urllib.parse.quote(key,safe='/'),safe_path(dest,rel))
            token=tree.findtext('s:NextContinuationToken',namespaces=ns)
            if not token:break
    print('Inspect local event schema next; download completion alone is not a prepared dataset.')

def inventory(args):
    root=Path(args.root).resolve();items=[]
    for p in sorted(root.rglob('*events.tsv')):
        d=pd.read_csv(p,sep='\t');entry={'path':str(p.relative_to(root)),'rows':len(d),'columns':{}}
        for c in d:
            vals=d[c].dropna().astype(str).unique()
            entry['columns'][c]={'unique_count':len(vals),'examples':vals[:20].tolist()}
        js=p.with_suffix('.json')
        if js.exists():entry['metadata']=json.loads(js.read_text())
        items.append(entry)
    dump(args.out,{'root':str(root),'event_files':items})
    if not items:raise ValueError('No events.tsv found. Extract archives first or supply a canonical trial manifest.')
    print('Inspect',args.out,'and confirm energy units and event timing from source documentation.')

def make_manifest(args):
    cfg=json.loads(Path(args.config).read_text());rows=[]
    for spec in cfg['sources']:
        root=Path(spec['root']).expanduser().resolve()
        if not spec.get('energy_column') or spec['energy_column'].startswith('SET_'):
            raise ValueError('Set verified energy_column for '+spec['name']+' using inventory; no label guessing.')
        if spec.get('units')!='J' or not spec.get('schema_verified'):
            raise ValueError('Confirm physical energy in J and set schema_verified=true for '+spec['name'])
        if not spec.get('cohort_verified'):
            raise ValueError('Confirm participant/cohort identity and internal-only Tiemann selection')
        found=sorted(root.glob(spec.get('events_glob','**/*events.tsv')))
        if not found:raise ValueError('No events files: '+str(root))
        for p in found:
            if any(s in str(p.relative_to(root)) for s in spec.get('exclude_path_contains',['derivatives'])):continue
            stem=p.name.removesuffix('_events.tsv');entities=dict(t.split('-',1) for t in stem.split('_') if '-' in t)
            if 'sub' not in entities:raise ValueError('Missing BIDS subject in '+str(p))
            subj='sub-'+entities['sub']
            if spec.get('subjects') and subj not in spec['subjects']:continue
            raws=[p.with_name(stem+'_eeg'+e) for e in ['.vhdr','.edf','.bdf','.set','.fif']]
            raws=[r for r in raws if r.exists()]
            if len(raws)!=1:raise ValueError(f'Need exactly one supported raw EEG beside {p}; found {raws}')
            events=pd.read_csv(p,sep='\t')
            for c,allowed in spec.get('event_filter',{}).items():events=events[events[c].astype(str).isin([str(x) for x in allowed])]
            col=spec['energy_column']
            if col not in events:raise ValueError(f'Missing {col} in {p}')
            energies=events[col]
            if spec.get('energy_map'):
                energies=energies.astype(str).map(spec['energy_map'])
            energies=pd.to_numeric(energies,errors='raise')*float(spec.get('energy_multiplier',1))
            if energies.isna().any():raise ValueError('Unmapped/missing energy; explicitly filter nonstimulus events first')
            for idx,row in events.iterrows():
                uid=spec.get('subject_map',{}).get(subj,spec['cohort']+'/'+subj)
                rows.append({'dataset':spec['name'],'subject':subj,'participant_uid':uid,'session':entities.get('ses','1'),'run':stem,
                    'raw_file':str(raws[0]),'onset_s':float(row[spec.get('onset_column','onset')])+float(spec.get('onset_offset_s',0)),
                    'energy_j':float(energies.loc[idx]),'trial_id':str(p.relative_to(root))+':'+str(idx),'source_url':spec['source_url']})
    d=pd.DataFrame(rows)
    if d.empty:raise ValueError('No selected trials')
    if not d.energy_j.between(.05,20).all():raise ValueError('Implausible energy: verify units/codes, do not clip them')
    if d.duplicated(['raw_file','onset_s']).any():raise ValueError('Repeated events/raw recordings across datasets')
    d.to_csv(args.out,index=False);print('Wrote',len(d),'labeled trials from',d.dataset.unique().tolist())

def prepare(args):
    import mne
    if args.lowpass<90 or args.sfreq<=2*args.lowpass:raise ValueError('Feature bands require lowpass >=90 and sfreq >2*lowpass')
    d=pd.read_csv(args.manifest);channels=args.channels.split(',');out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh preparation output directory')
    out.mkdir(parents=True,exist_ok=True);parts=[];meta=[];audit=[];expected_times=None
    for raw_file,g in d.groupby('raw_file',sort=True):
        raw=mne.io.read_raw(raw_file,preload=True,verbose='ERROR')
        # Case-insensitive exact matches only; no guessed electrode substitutions.
        lookup={c.casefold():c for c in raw.ch_names}
        if any(c.casefold() not in lookup for c in channels):raise ValueError('Missing common channel: '+raw_file)
        raw.pick([lookup[c.casefold()] for c in channels]);raw.rename_channels({lookup[c.casefold()]:c for c in channels});raw.reorder_channels(channels)
        if any(c in raw.info['bads'] for c in channels):raise ValueError('Required electrode marked bad; inspect recording: '+raw_file)
        raw.set_eeg_reference('average',projection=False,verbose='ERROR')
        if raw.info['sfreq']<2.2*args.lowpass:raise ValueError('Original sampling rate too low for requested shared band')
        raw.filter(1,args.lowpass,method='iir',iir_params={'order':4,'ftype':'butter'},verbose='ERROR')
        raw.resample(args.sfreq,verbose='ERROR')
        g=g.sort_values('onset_s').copy()
        samples=np.rint(g.onset_s.to_numpy()*args.sfreq).astype(int)+raw.first_samp
        if len(np.unique(samples))!=len(samples):raise ValueError('Overlapping/duplicate event sample indices')
        events=np.c_[samples,np.zeros(len(g),int),np.ones(len(g),int)]
        epochs=mne.Epochs(raw,events,event_id={'stimulus':1},tmin=-.2,tmax=.8,baseline=(-.2,0),preload=True,
            reject={'eeg':args.reject_uv*1e-6},reject_by_annotation=True,verbose='ERROR')
        # Offline poststimulus model, not prestimulus prediction. Avoid immediate stimulus artifact.
        epochs.crop(tmin=.02,tmax=.8);x=epochs.get_data(copy=True).astype(np.float32)*1e6
        if expected_times is not None and not np.array_equal(expected_times,epochs.times):raise ValueError('Inconsistent epoch time grid')
        expected_times=epochs.times.copy();parts.append(x);meta.append(g.iloc[epochs.selection].copy())
        audit.append({'raw_file':raw_file,'sha256':digest(raw_file),'input_trials':len(g),'retained_trials':len(x)})
    if not parts or sum(map(len,parts))==0:raise ValueError('No clean epochs')
    x=np.concatenate(parts);m=pd.concat(meta,ignore_index=True)
    # Exact duplicate waveform identity can expose accidental cohort copies.
    hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in x]
    if pd.Series(hashes).duplicated().any():raise ValueError('Duplicate waveforms found: resolve source overlap before splitting')
    np.save(out/'epochs.npy',x);m.to_csv(out/'trials.csv',index=False)
    dump(out/'provenance.json',{'synthetic':False,'manifest_sha256':digest(args.manifest),'channels':channels,'sfreq':args.sfreq,
        'times':expected_times.tolist(),'units':'microvolts','target':'energy_j','target_units':'J','audit':audit,
        'preprocessing':'common-channel average reference, 1-lowpass Hz IIR, resample, -0.2..0 baseline, +0.02..0.8s retained',
        'lowpass':args.lowpass,'rejection_uv':args.reject_uv})
    print('Prepared',x.shape,'from',m.dataset.unique().tolist())

def split(m,external=None):
    m=m.copy();m['split']='train';m['fold']=-1
    # Global UID assignment ensures repeated people never cross train/test or CV folds.
    membership=m.groupby('participant_uid').dataset.apply(lambda s:'+'.join(sorted(set(s))))
    for signature,ids in membership.groupby(membership):
        ids=sorted(ids.index,key=rank_id)
        if external and external in signature.split('+'):test=ids;dev=[]
        else:
            n=0 if external else max(2,int(np.ceil(.2*len(ids))))
            test=ids[:n];dev=ids[n:]
            if len(dev)<6:raise ValueError('Need >=6 development subjects per cohort signature')
        m.loc[m.participant_uid.isin(test),'split']='test'
        for i,uid in enumerate(dev):m.loc[m.participant_uid==uid,'fold']=i%3
    if m[m.split=='test'].empty:raise ValueError('No held-out subjects')
    assert not set(m[m.split=='train'].participant_uid)&set(m[m.split=='test'].participant_uid)
    return m

def features(x,times,sfreq,channels):
    freq,pow=welch(x,fs=sfreq,nperseg=min(x.shape[-1],128),axis=-1);blocks=[];names=[]
    for band,lo,hi in [('delta',1,4),('theta',4,8),('alpha',8,13),('beta',13,30),('gamma',30,90)]:
        mask=(freq>=lo)&(freq<hi)
        values=np.log(np.maximum(pow[:,:,mask].mean(axis=-1),1e-12));blocks.append(values);names.extend([f'{c}_{band}' for c in channels])
    for label,lo,hi,kind in [('n2',.15,.35,'min'),('p2',.25,.55,'max')]:
        idx=np.flatnonzero((times>=lo)&(times<=hi));segment=x[:,:,idx]
        peaks=np.argmin(segment,axis=-1) if kind=='min' else np.argmax(segment,axis=-1)
        amp=np.take_along_axis(segment,peaks[:,:,None],axis=-1)[:,:,0];lat=times[idx][peaks]
        blocks.extend([amp,lat]);names.extend([f'{c}_{label}_amp' for c in channels]+[f'{c}_{label}_lat' for c in channels])
    blocks.append(np.log(np.maximum(x.std(axis=-1),1e-6)));names.extend([c+'_log_std' for c in channels])
    return np.concatenate(blocks,axis=1),names

def tabular(family,p):
    pre=[SimpleImputer(strategy='median'),StandardScaler()]
    if family=='spline':pre += [SplineTransformer(n_knots=p['knots'],degree=3,extrapolation='linear'),Ridge(alpha=p['alpha'])]
    elif family=='extra_trees':pre += [ExtraTreesRegressor(n_estimators=400,min_samples_leaf=p['leaf'],max_features=.8,n_jobs=-1,random_state=SEED)]
    else:raise ValueError(family)
    return make_pipeline(*pre)

def score(y,p):return {'r2':float(r2_score(y,p)),'mae_j':float(mean_absolute_error(y,p)),'rmse_j':float(np.sqrt(mean_squared_error(y,p)))}
def ci(m,y,p):
    v=pd.DataFrame({'g':m.participant_uid.to_numpy(),'y':y,'p':p});v['y2']=v.y**2;v['err']=(v.y-v.p)**2
    a=v.groupby('g').agg(n=('y','size'),s=('y','sum'),ss=('y2','sum'),err=('err','sum')).to_numpy()
    rng=np.random.default_rng(SEED);b=a[rng.integers(0,len(a),(2000,len(a)))].sum(axis=1);den=b[:,2]-b[:,1]**2/b[:,0]
    vals=1-b[den>1e-12,3]/den[den>1e-12]
    return np.quantile(vals,[.025,.975]).tolist() if len(vals) else None

def net_init(nch,nt,sfreq,p):
    import torch
    from braindecode.models import ATCNet
    # Braindecode returns logits (no softmax); n_outputs=1 makes a scalar regression head.
    return ATCNet(n_chans=nch,n_outputs=1,n_times=nt,sfreq=sfreq,conv_block_n_filters=16,
        conv_block_kernel_length_1=32,conv_block_kernel_length_2=16,conv_block_pool_size_1=4,
        conv_block_pool_size_2=4,n_windows=3,tcn_depth=2,tcn_kernel_size=3,
        conv_block_dropout=p['dropout'],tcn_drop_prob=p['dropout'],max_norm_const=2.0)

def neural(x,y,m,tr,ev,sfreq,p,epochs,device,seed):
    import torch
    torch.manual_seed(seed);np.random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    # Inner participant split ONLY from the current fitting partition controls early stopping.
    ids=sorted(m.iloc[tr].participant_uid.unique(),key=rank_id);n=max(1,int(np.ceil(.15*len(ids))))
    val=np.array([i for i in tr if m.iloc[i].participant_uid in set(ids[:n])]);train=np.setdiff1d(tr,val)
    if len(ids)-n<2:raise ValueError('Too few inner training subjects')
    def fit_for(indices,iterations,validation=None):
        mu=x[indices].mean(axis=(0,2),keepdims=True);sd=np.maximum(x[indices].std(axis=(0,2),keepdims=True),1e-3)
        ym=float(y[indices].mean());ys=max(float(y[indices].std()),1e-3)
        net=net_init(x.shape[1],x.shape[2],sfreq,p).to(device);opt=torch.optim.AdamW(net.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
        rng=np.random.default_rng(seed);best=float('inf');best_epoch=1;stale=0
        for epoch in range(1,iterations+1):
            net.train()
            for batch in np.array_split(rng.permutation(indices),max(1,int(np.ceil(len(indices)/64)))):
                xb=torch.as_tensor((x[batch]-mu)/sd,dtype=torch.float32,device=device);yb=torch.as_tensor((y[batch]-ym)/ys,dtype=torch.float32,device=device)
                opt.zero_grad();out=net(xb).reshape(-1)
                if out.shape!=yb.shape:raise ValueError('Unexpected ATCNet output shape')
                loss=torch.nn.functional.mse_loss(out,yb)
                if not torch.isfinite(loss):raise ValueError('Nonfinite ATCNet loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            if validation is not None:
                pp=predict(net,validation,mu,sd,ym,ys);err=float(np.mean((y[validation]-pp)**2))
                if err<best-1e-6:best=err;best_epoch=epoch;stale=0
                else:stale+=1
                if stale>=12:break
        return net,mu,sd,ym,ys,best_epoch
    def predict(net,idx,mu,sd,ym,ys):
        net.eval();pp=[]
        with torch.no_grad():
            for b in np.array_split(idx,max(1,int(np.ceil(len(idx)/128)))):
                pp.extend((net(torch.as_tensor((x[b]-mu)/sd,dtype=torch.float32,device=device)).reshape(-1).cpu().numpy()*ys+ym).tolist())
        return np.asarray(pp)
    inner=fit_for(train,epochs,val);chosen=inner[-1];del inner
    # Refit all current fitting subjects for the selected epoch count. Never stop on outer validation/test.
    torch.manual_seed(seed);net,mu,sd,ym,ys,_=fit_for(tr,chosen)
    pp=predict(net,ev,mu,sd,ym,ys)
    state={'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'mu':mu,'sd':sd,'ym':ym,'ys':ys,'epochs':chosen,'params':p,'n_chans':x.shape[1],'n_times':x.shape[2],'sfreq':sfreq,'seed':seed}
    return pp,state

def load_prepared(folder, require_labels=True):
    root=Path(folder);x=np.load(root/'epochs.npy',mmap_mode='r');m=pd.read_csv(root/'trials.csv');p=json.loads((root/'provenance.json').read_text())
    if len(m)!=len(x) or not np.isfinite(x).all():raise ValueError('Invalid prepared data')
    if require_labels and ('energy_j' not in m or not np.isfinite(m.energy_j).all()):raise ValueError('Finite energy labels required')
    if m.duplicated(['dataset','trial_id']).any():raise ValueError('Duplicate trial keys')
    return x,m,p

def benchmark(args):
    x,m,prov=load_prepared(args.data)
    if prov.get('synthetic'):raise ValueError('Synthetic data are for smoke tests, never a benchmark')
    if not any(s.startswith('zhao') for s in m.dataset.unique()) or 'tiemann_munich' not in set(m.dataset):
        raise ValueError('This experiment REQUIRES both Zhao and Tiemann Munich. No Zhao-only fallback.')
    if args.external and args.external not in set(m.dataset):raise ValueError('Unknown external dataset')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new output directory; no stale training cache is accepted')
    out.mkdir(parents=True,exist_ok=True);m=split(m,args.external);m.to_csv(out/'splits.csv',index=False)
    tr=np.flatnonzero(m.split=='train');te=np.flatnonzero(m.split=='test');y=m.energy_j.to_numpy()
    f,names=features(x,np.array(prov['times']),prov['sfreq'],prov['channels'])
    grids={'spline':[{'knots':k,'alpha':a} for k in [3,5] for a in [10.,100.,1000.]],
        'extra_trees':[{'leaf':v} for v in [5,15,40]],
        'atcnet':[{'lr':lr,'dropout':dr,'weight_decay':wd} for lr,dr,wd in [(1e-3,.3,1e-3),(3e-4,.3,1e-2),(1e-3,.5,1e-2)]]}
    candidates={};oofs={};chosen={}
    for family in args.models:
        candidates[family]=[];best=float('inf')
        for params in grids[family]:
            oof=np.full(len(m),np.nan)
            for fold in range(3):
                a=tr[m.iloc[tr].fold.to_numpy()!=fold];b=tr[m.iloc[tr].fold.to_numpy()==fold]
                assert not set(m.iloc[a].participant_uid)&set(m.iloc[b].participant_uid)
                if family=='atcnet':pred,_=neural(x,y,m,a,b,prov['sfreq'],params,args.epochs,args.device,SEED)
                else:est=tabular(family,params);est.fit(f[a],y[a]);pred=est.predict(f[b])
                oof[b]=pred
            met=score(y[tr],oof[tr]);record={'params':params,'cv':met};candidates[family].append(record)
            print(family,params,met,flush=True)
            if met['rmse_j']<best:best=met['rmse_j'];chosen[family]=record;oofs[family]=oof
            dump(out/'development.json',{'candidates':candidates,'selected':chosen})
    blend=None
    if 'atcnet' in oofs and 'spline' in oofs:
        delta=oofs['atcnet'][tr]-oofs['spline'][tr];den=float(delta@delta)
        w=float(np.clip(delta@(y[tr]-oofs['spline'][tr])/den,0,1)) if den>1e-12 else .5
        blend={'atcnet_weight':w,'spline_weight':1-w,'note':'Fit using development out-of-fold predictions only; CV score after tuning is not an unbiased final score.'}
    dump(out/'selection_locked.json',{'chosen':chosen,'ensemble':blend,'target':'energy_j','code_sha256':digest(__file__)})
    preds={};models={}
    for family,record in chosen.items():
        if family=='atcnet':
            import torch
            # Same seed protocol as CV; seed ensembling can be a separately registered follow-up.
            pp,state=neural(x,y,m,tr,te,prov['sfreq'],record['params'],args.epochs,args.device,SEED)
            torch.save(state,out/'atcnet.pt');preds[family]=pp
        else:
            est=tabular(family,record['params']);est.fit(f[tr],y[tr]);preds[family]=est.predict(f[te]);models[family]=est
    if blend:preds['ensemble']=blend['atcnet_weight']*preds['atcnet']+blend['spline_weight']*preds['spline']
    means=m.iloc[tr].groupby('dataset').energy_j.mean();preds['training_mean']=m.iloc[te].dataset.map(means).fillna(y[tr].mean()).to_numpy()
    results={};rows=m.iloc[te].copy()
    for name,p in preds.items():
        results[name]={'pooled':score(y[te],p),'r2_ci95':ci(m.iloc[te],y[te],p),'by_dataset':{}}
        for ds in m.iloc[te].dataset.unique():
            mask=m.iloc[te].dataset.to_numpy()==ds;results[name]['by_dataset'][ds]=score(y[te][mask],p[mask])
        rows[name]=p
    rows.to_csv(out/'predictions.csv',index=False)
    joblib.dump({'models':models,'ensemble':blend,'feature_names':names,'provenance':prov},out/'tabular.joblib')
    dump(out/'results.json',{'target':'physical laser energy in J','heldout_dataset':args.external,'train_subjects':m.iloc[tr].participant_uid.nunique(),
        'test_subjects':m.iloc[te].participant_uid.nunique(),'train_datasets':m.iloc[tr].dataset.value_counts().to_dict(),'test_datasets':m.iloc[te].dataset.value_counts().to_dict(),
        'results':results,'code_sha256':digest(__file__),'data_sha256':digest(Path(args.data)/'epochs.npy'),'python':platform.python_version(),
        'interpretation':'Scores are study-protocol dependent; shared joules do not guarantee equivalent nociceptive exposure.'})
    print(json.dumps(results,indent=2))

def predict_command(args):
    x,m,p=load_prepared(args.data,require_labels=False);b=joblib.load(Path(args.models)/'tabular.joblib')
    for k in ['channels','sfreq','times','lowpass']:
        if p[k]!=b['provenance'][k]:raise ValueError('Preprocessing mismatch: '+k)
    f,names=features(x,np.array(p['times']),p['sfreq'],p['channels']);preds={n:est.predict(f) for n,est in b['models'].items()}
    path=Path(args.models)/'atcnet.pt'
    if path.exists():
        import torch
        state=torch.load(path,map_location='cpu',weights_only=False) # Load only this trusted, self-generated checkpoint.
        net=net_init(state['n_chans'],state['n_times'],state['sfreq'],state['params']).to(args.device);net.load_state_dict(state['state_dict']);net.eval();v=[]
        with torch.no_grad():
            for ix in np.array_split(np.arange(len(x)),max(1,int(np.ceil(len(x)/128)))):
                t=torch.as_tensor((x[ix]-state['mu'])/state['sd'],dtype=torch.float32,device=args.device)
                v.extend((net(t).reshape(-1).cpu().numpy()*state['ys']+state['ym']).tolist())
        preds['atcnet']=np.array(v)
    if b['ensemble']:preds['ensemble']=b['ensemble']['atcnet_weight']*preds['atcnet']+b['ensemble']['spline_weight']*preds['spline']
    ans=m[['dataset','subject','trial_id']].copy()
    for k,v in preds.items():ans[k]=v
    ans.to_csv(args.out,index=False)

def smoke(args):
    rng=np.random.default_rng(SEED);times=np.arange(.02,.804,1/250);x=rng.normal(size=(120,3,len(times))).astype('float32')
    m=pd.DataFrame({'dataset':['zhao_test']*60+['tiemann_munich']*60,'participant_uid':[f'p{i//5}' for i in range(120)],'energy_j':rng.uniform(1,4.5,120)})
    m=split(m);f,n=features(x,times,250,['Fz','Cz','Pz']);tr=np.flatnonzero(m.split=='train');te=np.flatnonzero(m.split=='test')
    for fam,p in [('spline',{'knots':3,'alpha':100}),('extra_trees',{'leaf':5})]:
        est=tabular(fam,p);est.fit(f[tr],m.energy_j.iloc[tr]);assert np.isfinite(est.predict(f[te])).all()
    # Strong isolation assertions: no person's rows/sessions cross partitions.
    assert m.groupby('participant_uid').split.nunique().max()==1
    assert m[m.split=='train'].groupby('participant_uid').fold.nunique().max()==1
    if args.neural:
        import torch
        pp,state=neural(x,m.energy_j.to_numpy(),m,tr,te,250,{'lr':1e-3,'dropout':.3,'weight_decay':1e-3},2,args.device,SEED)
        assert np.isfinite(pp).all()
        # Ensure scalar output has a usable gradient (a 1-class softmax would fail this test).
        net=net_init(3,len(times),250,state['params']);net.train();loss=net(torch.randn(8,3,len(times))).square().mean();loss.backward()
        assert any(v.grad is not None and v.grad.abs().sum()>0 for v in net.parameters())
    print('PASS synthetic pipeline checks'+(' including ATCNet forward/backward/train' if args.neural else '; neural and MNE paths NOT executed')+'. These are NOT research results.')

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('download');p.add_argument('--source',choices=['tiemann','ds005473','ds005293'],required=True);p.add_argument('--dest',required=True)
    p=sub.add_parser('inventory');p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('manifest');p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('prepare');p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--channels',default='Fz,Cz,Pz,C3,C4');p.add_argument('--sfreq',type=float,default=250);p.add_argument('--lowpass',type=float,default=95);p.add_argument('--reject-uv',type=float,default=150)
    p=sub.add_parser('benchmark');p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--models',nargs='+',choices=['spline','extra_trees','atcnet'],default=['spline','extra_trees','atcnet']);p.add_argument('--epochs',type=int,default=100);p.add_argument('--device',default='cpu');p.add_argument('--external',default=None)
    p=sub.add_parser('predict');p.add_argument('--data',required=True);p.add_argument('--models',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    p=sub.add_parser('smoke');p.add_argument('--neural',action='store_true');p.add_argument('--device',default='cpu')
    a=parser.parse_args();{'download':download,'inventory':inventory,'manifest':make_manifest,'prepare':prepare,'benchmark':benchmark,'predict':predict_command,'smoke':smoke}[a.cmd](a)
if __name__=='__main__':main()
