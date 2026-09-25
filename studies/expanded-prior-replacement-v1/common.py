import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
E=Path(__file__).resolve().parent
ROOT=E.parents[1]
SOURCE=ROOT/'data/collection-2026-09-25'
RUN=Path(os.environ.get('HPO_RUN_DIR',str(ROOT/'runs/expanded-prior-replacement-v1'))).resolve()
os.environ['HPO_RUN_DIR']=str(RUN)
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'examples'))
from collector import read,write,digest,key
from quantile_gp_numpy import QuantileGP,normal_rank_scores,log_ei
FEATURES=['log_learning_rate','depth_active','depth','leaves_active','log_leaves','lossguide','log_min_child_weight','subsample','column_tree','column_level','column_node','lambda_active','log_lambda','alpha_active','log_alpha','gamma_active','log_gamma','log_max_bin']
BINARY=[1,3,5,11,13,15]
METHODS=['gp_prior','gp_no_history','gp_prior_matched2','gp_ei_online','smac','tpe','random']
LABELS=['GP + historical prior','GP without history','GP + prior, matched starts','GP-EI, online kernel','SMAC (pool)','Optuna TPE (pool)','Random']
def seed(*x):return int(hashlib.sha256(json.dumps(x).encode()).hexdigest()[:8],16)
def encode(configs):
 out=[]
 for cf in configs:
  p=cf['params'];d=p['max_depth'];l=p['max_leaves']
  x=[np.log(p['learning_rate']/.01)/np.log(30),float(d>0),(d-2)/10 if d else 0.,float(l>0),np.log(l/8)/np.log(32) if l else 0.,float(p['grow_policy']=='lossguide'),np.log(p['min_child_weight']/1e-5)/np.log(1e7),(p['subsample']-.5)/.5]
  x.extend((1-p['colsample_by'+k])/.5 for k in ['tree','level','node'])
  for name,hi in [('reg_lambda',100),('reg_alpha',10),('gamma',10)]:
   v=p[name];x.extend([float(v>0),np.log(v/1e-5)/np.log(hi/1e-5) if v>0 else 0.])
  x.append(np.log2(p['max_bin']/128)/2);out.append(x)
 a=np.asarray(out);assert a.shape==(len(configs),18) and a.min()>-1e-10 and a.max()<1+1e-10
 return np.clip(a,0,1)
def kernel(theta,d,gradient=False):
 n=d.shape[-1];q=d/np.exp(2*theta[:n]);r=np.sqrt(5*q.sum(axis=2));e=np.exp(-r);amp=np.exp(theta[n]);K=amp*(1+r+r*r/3)*e
 if not gradient:return K
 g=amp*5/3*(1+r[:,:,None])*e[:,:,None]*q
 return K,np.concatenate([g,K[:,:,None]],axis=2)
def objective(theta,groups):
 from scipy.linalg import cho_solve
 value=0.;grad=np.zeros_like(theta)
 for d,second,w in groups:
  K,g=kernel(theta,d,True);n=len(K);noise=np.exp(theta[-1]);L=np.linalg.cholesky(K+(noise+1e-10)*np.eye(n));inv=cho_solve((L,True),np.eye(n),check_finite=False)
  value+=w*.5*(2*np.log(np.diag(L)).sum()+np.sum(inv*second)+n*np.log(2*np.pi))/n
  Q=.5*(inv-inv@second@inv)/n
  grad+=w*np.r_[np.einsum('ij,ijk->k',Q,g),noise*np.trace(Q)]
 return float(value),grad

def physical_cores():
 out=[];seen=set()
 for cpu in sorted(os.sched_getaffinity(0)):
  p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology');pair=((p/'physical_package_id').read_text(),(p/'core_id').read_text())
  if pair not in seen:out.append(cpu);seen.add(pair)
 return out[:16]
def init_worker(cpus):
 import multiprocessing as mp
 os.sched_setaffinity(0,{cpus[(mp.current_process()._identity[0]-1)%len(cpus)]})
def verify_freeze():
 f=read(E/'freeze.json')
 for p,h in f['hashes'].items():assert digest(ROOT/p)==h,p
 return f
