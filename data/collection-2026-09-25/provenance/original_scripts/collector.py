"""Bounded, resumable collection of real XGBoost evaluations; no HPO policy."""
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_k]='1'
from pathlib import Path
import sys,json,hashlib,time,platform,resource,traceback,fcntl,argparse,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
R=Path(__file__).resolve().parent; S=R.parent
sys.path.insert(0,str(S/'development/vendor'))
import numpy as np
import pandas as pd
import scipy
from scipy.stats import qmc
import xgboost as xgb
from sklearn.metrics import log_loss,root_mean_squared_error
ROUND_CAP=2000; PATIENCE=100; FIT_CPU_CAP=120; RUN_WALL_CAP=86400
CACHE={}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def read(p):return json.loads(Path(p).read_text())
def logspace(u,a,b):return float(np.exp(np.log(a)+u*np.log(b/a)))
def spike(u,hi):return 0. if u<.25 else logspace((u-.25)/.75,1e-5,hi)

def matrices(sp,max_bin,threads=1):
    if CACHE and next(iter(CACHE))[0] != sp['uid']: CACHE.clear()
    ck=(sp['uid'],max_bin,threads)
    if ck in CACHE:return CACHE[ck],False
    d=Path(sp['data_dir']);frames={n:pd.read_pickle(d/f'{n}_X.pkl') for n in ['train','validation']};ys={n:np.load(d/f'{n}_y.npy') for n in frames}
    cats=sp['categoricals'];schema={c:sorted(frames['train'][c].dropna().unique().tolist()) for c in cats};aa={};meta={}
    for n,frame in frames.items():
        aa[n]=np.column_stack([pd.Categorical(frame[c].where(frame[c].isin(schema[c])),categories=schema[c]).codes.astype(float) if c in cats else frame[c].to_numpy(float) for c in sp['features']]).astype(np.float32)
        for j,c in enumerate(sp['features']):
            if c in cats:aa[n][aa[n][:,j]<0,j]=np.nan
        meta[n+'_unknown_category_cells']=int(sum((~frame[c].isin(schema[c])&frame[c].notna()).sum() for c in cats))
    ft=['c' if c in cats else 'q' for c in sp['features']]
    dt=xgb.QuantileDMatrix(aa['train'],label=ys['train'],feature_types=ft,enable_categorical=True,max_bin=max_bin,nthread=threads)
    dv=xgb.QuantileDMatrix(aa['validation'],label=ys['validation'],feature_types=ft,enable_categorical=True,max_bin=max_bin,nthread=threads,ref=dt)
    if sp['task']=='regression':baseline=float(root_mean_squared_error(ys['validation'],np.full(len(ys['validation']),ys['train'].mean())))
    else:
        pr=np.bincount(ys['train'].astype(int),minlength=sp['n_classes'])/len(ys['train']);baseline=float(log_loss(ys['validation'],np.tile(pr,(len(ys['validation']),1)),labels=np.arange(sp['n_classes'])))
    CACHE[ck]=(dt,dv,ys,baseline,meta);return CACHE[ck],True

class Guard(xgb.callback.TrainingCallback):
    def __init__(self):self.t=time.process_time();self.w=time.perf_counter();self.cpu=[];self.wall=[];self.stopped=None
    def after_iteration(self,model,epoch,evals_log):
        self.cpu.append(time.process_time()-self.t);self.wall.append(time.perf_counter()-self.w)
        if self.cpu[-1]>FIT_CPU_CAP:self.stopped='cpu_cap';return True
        if (R/'STOP').exists():self.stopped='stop_requested';return True
        return False

def structure(bst):
    # Read structured model arrays, avoiding expensive human-readable per-node dumps.
    trees=json.loads(bytes(bst.save_raw(raw_format='json')))['learner']['gradient_booster']['model']['trees']
    leaves=[];maxdepth=[];avgdepth=[];leaf_abs=[];total_depth=0;total_leaf=0
    for tr in trees:
        left=tr['left_children'];right=tr['right_children'];weight=tr['split_conditions'];stack=[(0,0)];ds=[];ws=[]
        while stack:
            i,depth=stack.pop()
            if left[i]==-1:ds.append(depth);ws.append(abs(weight[i]))
            else:stack.extend([(left[i],depth+1),(right[i],depth+1)])
        leaves.append(len(ds));maxdepth.append(max(ds));avgdepth.append(float(np.mean(ds)));leaf_abs.extend(ws);total_depth+=sum(ds);total_leaf+=len(ds)
    return dict(n_trees=len(trees),total_leaves=sum(leaves),leaves_per_tree=leaves,max_depth_per_tree=maxdepth,mean_leaf_depth_per_tree=avgdepth,mean_leaf_depth=total_depth/total_leaf,max_realized_depth=max(maxdepth),mean_abs_leaf_output=float(np.mean(leaf_abs)),max_abs_leaf_output=float(max(leaf_abs)))

def evaluate(sp,c,identity,group='records',threads=1,suffix='',round_cap=ROUND_CAP):
    rid=sp['uid']+'__'+c['cid']+suffix;p=R/group/(rid+'.json')
    if p.exists():
        z=read(p);assert z['identity']==identity and z['params_sampled']==c['params'];return z
    t=time.process_time();w=time.perf_counter();(dt,dv,ys,baseline,meta),cold=matrices(sp,c['params']['max_bin'],threads);prep=time.process_time()-t
    params=dict(c['params'],tree_method='hist',device='cpu',nthread=threads,seed=20260924+sp['did'],verbosity=0,validate_parameters=True)
    if sp['task']=='regression':params.update(objective='reg:squarederror',eval_metric='rmse');metric='rmse'
    elif sp['n_classes']==2:params.update(objective='binary:logistic',eval_metric='logloss');metric='logloss'
    else:params.update(objective='multi:softprob',num_class=sp['n_classes'],eval_metric='mlogloss');metric='mlogloss'
    history={};guard=Guard();ft=time.process_time();fw=time.perf_counter()
    try:
        bst=xgb.train(params,dt,num_boost_round=round_cap,evals=[(dt,'train'),(dv,'validation')],evals_result=history,early_stopping_rounds=PATIENCE,verbose_eval=False,callbacks=[guard])
        traincpu=time.process_time()-ft;trainwall=time.perf_counter()-fw
        train=np.asarray(history['train'][metric]);val=np.asarray(history['validation'][metric]);best=int(bst.best_iteration);trained=int(bst.num_boosted_rounds())
        assert best==int(np.argmin(val)) and len(val)==trained and np.isfinite(val).all() and np.isfinite(train).all()
        selected=bst[:best+1];pred=selected.predict(dv)
        independent=float(root_mean_squared_error(ys['validation'],pred)) if sp['task']=='regression' else float(log_loss(ys['validation'],pred,labels=np.arange(sp['n_classes'])))
        assert np.isclose(independent,val[best],rtol=2e-5,atol=2e-6),(independent,val[best])
        st=structure(selected);mt=R/group/(rid+'.ubj.zlib');import zlib;mt.write_bytes(zlib.compress(bytes(selected.save_raw(raw_format='ubj')),level=1))
        cp=R/group/(rid+'.npz');np.savez_compressed(cp,train_loss=train,validation_loss=val,round_cpu=np.asarray(guard.cpu),round_wall=np.asarray(guard.wall),validation_prediction=pred)
        stopped=guard.stopped or ('round_cap' if trained==round_cap else 'early_stopping')
        z=dict(identity=identity,schema_version=1,record_id=rid,uid=sp['uid'],dataset=sp['dataset'],family=sp['family'],cid=c['cid'],design_group=c['group'],size_mode=c['size_mode'],column_mode=c['column_mode'],params_sampled=c['params'],params_effective=params,seed=params['seed'],status='censored' if guard.stopped else 'complete',stop_reason=stopped,metric=metric,validation_loss=float(val[best]),validation_loss_independent=independent,train_loss_at_best=float(train[best]),validation_baseline=baseline,best_iteration=best,selected_rounds=best+1,trained_rounds=trained,round_cap=round_cap,patience=PATIENCE,train_cpu_seconds=traincpu,train_wall_seconds=trainwall,preparation_cpu_seconds=prep,cache_miss=cold,structure=st,model_bytes=mt.stat().st_size,model_format='ubj_zlib_level1',model_sha256=digest(mt),curves_sha256=digest(cp),preprocessing=meta)
    except Exception as e:
        z=dict(identity=identity,record_id=rid,uid=sp['uid'],cid=c['cid'],params_sampled=c['params'],status='failed',error=repr(e),traceback=traceback.format_exc())
    z.update(total_cpu_seconds=time.process_time()-t,total_wall_seconds=time.perf_counter()-w,worker_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,pid=os.getpid())
    write(p,z);return z
