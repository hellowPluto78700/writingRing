from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import copy, hashlib, math, os, random
import numpy as np, pandas as pd, torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import snntorch as snn
from snntorch import surrogate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from snn.accel_reconstruction_eval.datasets import load_acceleration_data

EXPERIMENT_ID='experiment_3_0_1_single_tau_objective_comparison'
SHIFTS=(2,3,4,5,6,7); OBJECTIVES=('timestep_ce','relative10_sequence_ce','fixed250_sequence_ce'); SEEDS=(11,23,101)
SPLIT_SEED=12345; WIDTHS=(128,128,64); TAU_MEM_MS=22.54; THRESHOLD=0.5; SURROGATE_SLOPE=25.; RESET='subtract'
BATCH_SIZE=128; EPOCHS=100; LR=1e-3; N_REL=10; FIXED_MS=250.; NWORKERS=0; LABELS=('A','B','C','D','E','X','G','H','I','J','K','L')
EXPECTED_RUNS=len(SHIFTS)*len(OBJECTIVES)*len(SEEDS)

def dseed(seed,*parts): return int.from_bytes(hashlib.sha256('|'.join(map(str,(seed,*parts))).encode()).digest()[:4],'little')
def seed_all(seed):
    os.environ['PYTHONHASHSEED']=str(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    try: torch.use_deterministic_algorithms(True,warn_only=True)
    except TypeError: torch.use_deterministic_algorithms(True)
def alpha(shift): return 1.-2.**(-int(shift))
def tau_ms(shift,fs): return -(1000./fs)/math.log(alpha(shift))

def tau_table(fs): return pd.DataFrame([{'shift':s,'alpha':alpha(s),'tau_syn_ms':tau_ms(s,fs)} for s in SHIFTS])

@dataclass
class Data:
    Xtr:np.ndarray; ytr:np.ndarray; ltr:np.ndarray; Xva:np.ndarray; yva:np.ndarray; lva:np.ndarray
    Xte:np.ndarray; yte:np.ndarray; lte:np.ndarray; labels:tuple; fs:float; T:int; bin_steps:int; n_bins:int; split:dict

@dataclass(frozen=True)
class Config:
    repo_root:Path; results_dir:Path; device:str='cpu'; epochs:int=EPOCHS; batch_size:int=BATCH_SIZE; resume:bool=True; threads:int=1

def prepare_data(repo_root:Path)->Data:
    roots=[repo_root/'outputs/action0_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded',repo_root/'outputs/action1_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded']
    data=load_acceleration_data(roots,repository_root=repo_root,require_reconstruction=False)
    fsvals={float(m.sampling_rate_hz) for m in data.producer_metadatas}; assert len(fsvals)==1; fs=fsvals.pop(); assert np.isclose(fs,64.)
    for root,m in zip(data.padded_roots,data.producer_metadatas,strict=True):
        raw=m.raw; assert raw.get('event_representation')=='unsigned'; assert raw.get('event_feature_schema')=='custom_wavelet_polarity_split_abs_events_v1'; assert raw.get('event_channel_count')==30; assert m.channel_count==36
    rows=[]; keep=set(LABELS)
    for pi,p in enumerate(data.packages):
        for si,label in enumerate(p.labels.astype(str)):
            if label in keep: rows.append((pi,si,str(p.user),str(label),int(p.valid_lengths[si]),int(p.padded_spike_imu.shape[1])))
    mf=pd.DataFrame(rows,columns=['pi','si','user','label','valid','pad']); labs=tuple(sorted(mf.label.unique())); cmap={x:i for i,x in enumerate(labs)}; mf['y']=mf.label.map(cmap).astype(int)
    users=np.array(sorted(mf.user.unique()),dtype=object); rng=np.random.default_rng(dseed(SPLIT_SEED,'split')); rng.shuffle(users); nt=round(.7*len(users)); nv=round(.15*len(users)); tr=set(users[:nt]); va=set(users[nt:nt+nv]); te=set(users[nt+nv:])
    parts=[mf[mf.user.isin(u)].reset_index(drop=True) for u in (tr,va,te)]; T0=int(mf.pad.max()); bs=int(round(FIXED_MS*fs/1000)); nb=math.ceil(T0/bs); T=nb*bs; assert bs==16
    def build(df):
        out=np.zeros((len(df),T,30),np.float32)
        for i,r in enumerate(df.itertuples()):
            x=np.asarray(data.packages[r.pi].padded_spike_imu[r.si,:,:30],np.float32); n=min(len(x),T,r.valid); out[i,:n]=x[:n]
        return out
    arr=[]
    for df in parts: arr.extend([build(df),df.y.to_numpy(np.int64),np.minimum(df.valid.to_numpy(np.int64),T)])
    return Data(*arr,labs,fs,T,bs,nb,{'train_users':tuple(sorted(tr)),'val_users':tuple(sorted(va)),'test_users':tuple(sorted(te))})

def loader(X,y,L,batch,shuffle,seed):
    ds=TensorDataset(torch.from_numpy(X),torch.from_numpy(y),torch.from_numpy(L)); g=torch.Generator().manual_seed(seed)
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=NWORKERS,generator=g)
def mask(L,T): return torch.arange(T,device=L.device)[None,:] < L[:,None]
def fixed_counts(s,L,steps):
    B,T,D=s.shape; z=s*mask(L,T).to(s.dtype).unsqueeze(-1); n=math.ceil(T/steps); z=F.pad(z,(0,0,0,n*steps-T)); return z.reshape(B,n,steps,D).sum(2)
def relative_counts(s,L,n=N_REL):
    B,T,D=s.shape; p=torch.arange(T,device=s.device)[None,:].expand(B,T); den=L.clamp_min(1)[:,None]; good=p<den; idx=torch.div(p*n,den,rounding_mode='floor').clamp(max=n-1); out=torch.zeros(B,n,D,device=s.device,dtype=s.dtype); out.scatter_add_(1,idx.unsqueeze(-1).expand(-1,-1,D),s*good.to(s.dtype).unsqueeze(-1)); return out

class Net(nn.Module):
    def __init__(self,shift,obj,nclass,T,fs,bin_steps):
        super().__init__(); h1,h2,d=WIDTHS; a=alpha(shift); b=math.exp(-(1000./fs)/TAU_MEM_MS); sg=surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)
        mk=lambda w:snn.Synaptic(alpha=torch.full((w,),a),beta=b,threshold=THRESHOLD,spike_grad=sg,reset_mechanism=RESET)
        self.f1=nn.Linear(30,h1,bias=False); self.l1=mk(h1); self.f2=nn.Linear(h1,h2,bias=False); self.l2=mk(h2); self.f3=nn.Linear(h2,d,bias=False); self.l3=mk(d); self.obj=obj; self.T=T; self.steps=bin_steps
        dim=d if obj=='timestep_ce' else d*N_REL if obj=='relative10_sequence_ce' else d*math.ceil(T/bin_steps); self.head=nn.Linear(dim,nclass)
    def features(self,x,L,stats=False):
        B,T,_=x.shape; h1,h2,d=WIDTHS; z1=torch.zeros(B,h1,device=x.device); m1=z1.clone(); z2=torch.zeros(B,h2,device=x.device); m2=z2.clone(); z3=torch.zeros(B,d,device=x.device); m3=z3.clone(); seq=[]; sums=torch.zeros(3,device=x.device); v=L.float().sum()
        for t in range(T):
            s1,z1,m1=self.l1(self.f1(x[:,t]),z1,m1); s2,z2,m2=self.l2(self.f2(s1),z2,m2); s3,z3,m3=self.l3(self.f3(s2),z3,m3); seq.append(s3)
            if stats:
                q=(t<L).float().unsqueeze(-1); sums+=torch.stack([(s1*q).sum(),(s2*q).sum(),(s3*q).sum()])
        seq=torch.stack(seq,1); rates=None if not stats else sums/torch.tensor([v*h1,v*h2,v*d],device=x.device).clamp_min(1)
        return seq,rates
    def loss_logits(self,s,L,y):
        B,T,_=s.shape
        if self.obj=='timestep_ce':
            lg=self.head(s); q=mask(L,T); loss=F.cross_entropy(lg[q],y[:,None].expand(B,T)[q]); w=q.float().unsqueeze(-1); return loss,(lg*w).sum(1)/w.sum(1).clamp_min(1)
        z=relative_counts(s,L) if self.obj=='relative10_sequence_ce' else fixed_counts(s,L,self.steps); lg=self.head(z.flatten(1)); return F.cross_entropy(lg,y),lg
    def rep(self,s,L): return fixed_counts(s,L,self.steps).flatten(1)

def metrics(y,p): return {'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro',zero_division=0)}
def evaluate(model,ld,dev,stats=False,rep=False):
    model.eval(); ys=[]; ps=[]; losses=[]; reps=[]; rates=[]
    with torch.no_grad():
        for X,y,L in ld:
            X,y,L=X.to(dev),y.to(dev),L.to(dev); s,r=model.features(X,L,stats); loss,lg=model.loss_logits(s,L,y); ys.append(y.cpu().numpy()); ps.append(lg.argmax(1).cpu().numpy()); losses.append(loss.item()*len(y));
            if rep: reps.append(model.rep(s,L).cpu().numpy())
            if r is not None: rates.append((r.cpu().numpy(),len(y)))
    yt=np.concatenate(ys); out=metrics(yt,np.concatenate(ps)); out['loss']=sum(losses)/len(yt)
    if rates: out['rates']=sum(r*n for r,n in rates)/sum(n for _,n in rates)
    if rep: out['rep']=np.concatenate(reps); out['y']=yt
    return out

def probe(tr,va,te,seed):
    sc=StandardScaler().fit(tr['rep']); A,B,C=map(sc.transform,(tr['rep'],va['rep'],te['rep'])); best=None
    for c in (1e-3,1e-2,1e-1,1.,10.):
        m=LogisticRegression(C=c,max_iter=5000,solver='lbfgs',random_state=seed).fit(A,tr['y']); ba=balanced_accuracy_score(va['y'],m.predict(B));
        if best is None or ba>best[0]: best=(ba,c,m)
    ba,c,m=best; mm=metrics(te['y'],m.predict(C)); return {'probe_C':c,'probe_val_balanced_accuracy':ba,'probe_test_balanced_accuracy':mm['balanced_accuracy'],'probe_test_macro_f1':mm['macro_f1'],'probe_test_accuracy':mm['accuracy']}

def run_one(shift,obj,seed,data:Data,cfg:Config):
    path=cfg.results_dir/'checkpoints'/f'shift{shift}__{obj}__seed{seed}.pt'
    if cfg.resume and path.exists():
        p=torch.load(path,map_location='cpu',weights_only=False); return p['result']
    torch.set_num_threads(cfg.threads); dev=torch.device(cfg.device); seed_all(dseed(seed,'backbone',shift)); model=Net(shift,obj,len(data.labels),data.T,data.fs,data.bin_steps).to(dev); opt=torch.optim.Adam(model.parameters(),lr=LR)
    args=[(data.Xtr,data.ytr,data.ltr),(data.Xva,data.yva,data.lva),(data.Xte,data.yte,data.lte)]; tr=loader(*args[0],cfg.batch_size,True,dseed(seed,'train')); ev=[loader(*a,cfg.batch_size,False,dseed(seed,i)) for i,a in enumerate(args)]
    best=(-1.,float('inf')); state=None; epoch_best=0; hist=[]
    for ep in range(1,cfg.epochs+1):
        model.train(); yt=[]; yp=[]; ls=0
        for X,y,L in tr:
            X,y,L=X.to(dev),y.to(dev),L.to(dev); opt.zero_grad(set_to_none=True); s,_=model.features(X,L); loss,lg=model.loss_logits(s,L,y); loss.backward(); opt.step(); ls+=loss.item()*len(y); yt.append(y.cpu().numpy()); yp.append(lg.detach().argmax(1).cpu().numpy())
        tm=metrics(np.concatenate(yt),np.concatenate(yp)); vm=evaluate(model,ev[1],dev); hist.append({'epoch':ep,'train_loss':ls/len(data.ytr),'train_balanced_accuracy':tm['balanced_accuracy'],'val_loss':vm['loss'],'val_balanced_accuracy':vm['balanced_accuracy']}); key=(vm['balanced_accuracy'],-vm['loss'])
        if key>(best[0],-best[1]): best=(vm['balanced_accuracy'],vm['loss']); state=copy.deepcopy(model.state_dict()); epoch_best=ep
    model.load_state_dict(state); final=[evaluate(model,x,dev,True,True) for x in ev]; pr=probe(*final,dseed(seed,'probe',shift,obj)); zr=evaluate(model,loader(np.zeros((1,data.T,30),np.float32),np.zeros(1,np.int64),np.array([data.T],np.int64),1,False,0),dev,True)['rates']
    names=('train','val','test'); out={'shift':shift,'tau_syn_ms':tau_ms(shift,data.fs),'objective':obj,'seed':seed,'best_epoch':epoch_best,'history':hist,**pr}
    for name,m in zip(names,final):
        for k in ('loss','accuracy','balanced_accuracy','macro_f1'): out[f'{name}_{k}']=m[k]
    out['train_test_ba_gap']=out['train_balanced_accuracy']-out['test_balanced_accuracy']; out.update({f'test_l{i+1}_firing_rate':final[2]['rates'][i] for i in range(3)}); out.update({f'zero_l{i+1}_firing_rate':zr[i] for i in range(3)})
    path.parent.mkdir(parents=True,exist_ok=True); torch.save({'experiment_id':EXPERIMENT_ID,'result':out,'state_dict':state},path); return out

def run_sweep(data:Data,cfg:Config,n_jobs=1):
    specs=[(s,o,d) for s in SHIFTS for o in OBJECTIVES for d in SEEDS]
    if n_jobs>1 and cfg.device=='cpu':
        from joblib import Parallel, delayed, parallel_config
        with parallel_config(backend='loky',n_jobs=n_jobs,inner_max_num_threads=cfg.threads): rs=Parallel()(delayed(run_one)(s,o,d,data,cfg) for s,o,d in specs)
    else: rs=[run_one(s,o,d,data,cfg) for s,o,d in specs]
    rows=[{k:v for k,v in r.items() if k!='history'} for r in rs]; df=pd.DataFrame(rows).sort_values(['objective','shift','seed']); cfg.results_dir.mkdir(parents=True,exist_ok=True); df.to_csv(cfg.results_dir/'experiment_3_0_1_results.csv',index=False)
    pd.DataFrame([{'shift':r['shift'],'objective':r['objective'],'seed':r['seed'],**h} for r in rs for h in r['history']]).to_csv(cfg.results_dir/'experiment_3_0_1_history.csv',index=False); return df

def aggregate(df):
    cols=['test_balanced_accuracy','test_macro_f1','test_accuracy','probe_test_balanced_accuracy','probe_test_macro_f1','val_balanced_accuracy','train_test_ba_gap','best_epoch','test_l1_firing_rate','test_l2_firing_rate','test_l3_firing_rate']
    g=df.groupby(['objective','shift','tau_syn_ms'],as_index=False)[cols]
    mean=g.mean().rename(columns={c:'mean_'+c for c in cols}); std=g.std(ddof=1).rename(columns={c:'sd_'+c for c in cols})
    return mean.merge(std,on=['objective','shift','tau_syn_ms'])
