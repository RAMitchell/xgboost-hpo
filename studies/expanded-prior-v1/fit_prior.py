"""Fit mean, scale and ARD kernel exclusively from the 50 source families."""
from common import *
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize

def fit():
 if (E/'prior/model.json').exists():return
 start=time.process_time();wall=time.perf_counter();(E/'prior').mkdir(exist_ok=True)
 table=pd.read_parquet(SOURCE/'evaluations.parquet');configs=read(SOURCE/'configurations.json');X=encode(configs);ds=read(SOURCE/'datasets.json')
 mass=np.zeros(len(X));target=np.zeros(len(X));tasks=[]
 for sp in ds:
  rows=table[(table.uid==sp['uid']) & (table.status=='complete')].sort_values('cid')
  idx=np.array([int(cid.split('_')[-1]) for cid in rows.cid]);y=normal_rank_scores(rows.validation_loss.to_numpy());w=1/len(ds)
  mass[idx]+=w;target[idx]+=w*y;tasks.append((idx,y,w))
 target/=mass;weight=mass/mass.sum()*200
 params=dict(objective='reg:squarederror',max_depth=3,eta=.05,tree_method='hist',max_bin=128,nthread=1,min_child_weight=1,reg_lambda=1,reg_alpha=0,gamma=0,subsample=1.,colsample_bytree=1.)
 dm=xgb.DMatrix(X,label=target,weight=weight,nthread=1)
 mean_model=xgb.train(dict(params,base_score=0.,seed=9025),dm,num_boost_round=200);m=mean_model.predict(dm).astype(float);mean_model.save_model(E/'prior/mean.ubj')
 second=np.zeros(len(X))
 for idx,y,w in tasks:second[idx]+=w*(y-m[idx])**2
 logs=np.log(np.maximum(.0025,second/mass));dm.set_label(logs)
 scale_model=xgb.train(dict(params,base_score=float(np.average(logs,weights=weight)),seed=9026),dm,num_boost_round=200);s=np.maximum(.05,np.exp(np.clip(scale_model.predict(dm).astype(float),-12,12)/2));scale_model.save_model(E/'prior/scale.ubj')
 buckets={}
 for idx,y,w in tasks:
  k=tuple(idx);r=(y-m[idx])/s[idx]
  if k not in buckets:buckets[k]=[np.zeros((len(idx),len(idx))),0.]
  buckets[k][0]+=w*np.outer(r,r);buckets[k][1]+=w
 groups=[]
 for ids,(mom,w) in buckets.items():
  xx=X[list(ids)];groups.append(((xx[:,None]-xx[None,:])**2,mom/w,w))
 bounds=np.log([[.01,100.]]*18+[[1e-3,1e3],[1e-6,1.]])
 traces=[]
 for ell,noise in [(.2,1e-6),(.5,.01),(1.,.1)]:
  opt=minimize(lambda th:objective(th,groups),np.log([ell]*18+[1.,noise]),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=300,ftol=1e-10,gtol=1e-6))
  traces.append(dict(value=float(opt.fun),theta=opt.x.tolist(),success=bool(opt.success),message=str(opt.message),iterations=int(opt.nit)))
 converged=[z for z in traces if z['success']];assert converged,'No converged historical kernel fit'
 theta=np.array(min(converged,key=lambda z:z['value'])['theta']);Q=encode(read(E/'configurations.json'));qd=xgb.DMatrix(Q,nthread=1)
 mean=mean_model.predict(qd).astype(float);scale=np.maximum(.05,np.exp(np.clip(scale_model.predict(qd).astype(float),-12,12)/2));K=kernel(theta,(Q[:,None]-Q[None,:])**2)*scale[:,None]*scale[None,:]+1e-10*np.eye(len(Q));noise=scale**2*np.exp(theta[-1]);np.linalg.cholesky(K)
 np.savez_compressed(E/'prior/candidate_prior.npz',mean=mean,covariance=K,noise_variance=noise)
 model=dict(source_families=sorted(sp['family'] for sp in ds),source_datasets=len(ds),source_observations=sum(len(y) for _,y,_ in tasks),distinct_configurations=len(X),features=FEATURES,mean_training_predictions=m.tolist(),scale_training_predictions=s.tolist(),logtheta=theta.tolist(),lengthscales=dict(zip(FEATURES,np.exp(theta[:18]).tolist())),amplitude=float(np.exp(theta[-2])),noise_variance=float(np.exp(theta[-1])),restarts=traces,boundaries=np.flatnonzero(np.any(np.isclose(theta[:,None],bounds,atol=1e-4),axis=1)).tolist(),parameters=params,mean_scale_boosting_rounds=200,total_fit_weight=200,cpu_seconds=time.process_time()-start,wall_seconds=time.perf_counter()-wall,historical_xgb_fits=2,kernel_starts=3,target_losses_read=0)
 write(E/'prior/model.json',model)
 print(json.dumps({k:model[k] for k in ['source_observations','distinct_configurations','cpu_seconds','boundaries']}),flush=True)
if __name__=='__main__':fit()
