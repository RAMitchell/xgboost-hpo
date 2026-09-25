"""Replay frozen policies: only queried losses are passed to each optimizer."""
from common import *
from baselines import PoolBaseline
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp

def task(job):
 sp,method,rep=job;p=RUN/'trajectories'/f"{sp['uid']}__{method}__{rep}.json"
 if p.exists():return read(p)
 start=time.process_time();wall=time.perf_counter();configs=read(E/'configurations.json');X=encode(configs);seed_=seed('optimizer',sp['did'],rep)
 Y=read(RUN/'outcomes.json')[sp['uid']];truth=np.asarray(Y['objective']);status=Y['status'];uniform=np.random.default_rng(seed('shared-initial',sp['did'],rep)).permutation(len(X)).tolist()
 gp=None
 if method.startswith('gp_') and method!='gp_ei_online':
  if method=='gp_no_history':
   K=kernel(np.r_[np.zeros(18),0.,np.log(.1)],(X[:,None]-X[None,:])**2)+1e-10*np.eye(len(X));gp=QuantileGP(np.zeros(len(X)),K,.1,seed=seed_)
  else:
   with np.load(E/'prior/candidate_prior.npz') as a:gp=QuantileGP(a['mean'],a['covariance'],a['noise_variance'],seed=seed_)
 else:opt=PoolBaseline(method,X,seed_)
 seen=[];values=[];cost=[];failures=[]
 for step in range(32):
  if gp:
   # Matched control uses exactly the first two shared uniform proposals.
   if method=='gp_prior_matched2' and step<2:cid=uniform[step];gp.pending=cid
   else:cid=gp.ask()
  else:
   starts=10 if method in ['smac','tpe'] else 5 if method=='gp_ei_online' else 0
   cid=uniform[step] if step<starts else opt.ask()
  assert cid not in seen;v=float(truth[cid])
  if gp:gp.tell(cid,v)
  else:opt.tell(cid,v)
  seen.append(int(cid));values.append(v);cost.append(Y['cpu'][cid]);failures.append(status[cid]!='complete')
 z=dict(uid=sp['uid'],family=sp['family'],problem_type=sp['problem_type'],method=method,rep=rep,seed=seed_,ids=seen,values=values,best=np.minimum.accumulate(values).tolist(),objective_cpu=np.cumsum(cost).tolist(),noncomplete_queries=int(sum(failures)),cpu_seconds=time.process_time()-start,wall_seconds=time.perf_counter()-wall)
 write(p,z);return z

def build_table():
 datasets=read(RUN/'datasets.json');configs=read(E/'configurations.json');out={};audit=[]
 for sp in datasets:
  records=[read(RUN/'records'/(sp['uid']+'__'+c['cid']+'.json')) for c in configs]
  valid=[z['validation_loss'] for z in records if z['status']=='complete'];assert len(valid)>=2
  # Only reporting/replay oracle constructs this table. Policies never receive it.
  lo=min(valid);hi=max(valid);penalty=hi+max(hi-lo,abs(hi)*.01,1e-12)
  objective=[]
  for z,c in zip(records,configs):
   assert z['params_sampled']==c['params']
   if z['status']!='failed':
    p=RUN/'records'/(z['record_id']+'.npz');assert digest(p)==z['curves_sha256']
    with np.load(p) as a:assert len(a['validation_loss'])==z['trained_rounds'] and np.isclose(a['validation_loss'].min(),z['validation_loss'])
   objective.append(z['validation_loss'] if z['status']=='complete' else penalty)
  out[sp['uid']]=dict(objective=objective,status=[z['status'] for z in records],cpu=[z['total_cpu_seconds'] for z in records],minimum=lo,maximum=hi,penalty=penalty)
  audit.append(dict(uid=sp['uid'],complete=len(valid),censored=sum(z['status']=='censored' for z in records),failed=sum(z['status']=='failed' for z in records)))
 write(RUN/'outcomes.json',out);write(RUN/'outcome_audit.json',audit)

def main():
 verify_freeze();assert (RUN/'collection.json').exists();(RUN/'trajectories').mkdir(exist_ok=True);build_table();cpus=read(E/'design.json')['cpus'];os.sched_setaffinity(0,set(cpus));start=time.perf_counter();results=[]
 jobs=[(sp,m,r) for sp in read(RUN/'datasets.json') for m in METHODS for r in range(10)]
 # Expensive model-fitting baselines first to avoid a long tail.
 jobs.sort(key=lambda j:0 if j[1] in ['smac','gp_ei_online'] else 1)
 with ProcessPoolExecutor(len(cpus),mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(cpus,)) as pool:
  for future in as_completed([pool.submit(task,j) for j in jobs]):
   z=future.result();results.append(z)
   if len(results)%50==0:write(RUN/'replay_progress.json',dict(completed=len(results),expected=len(jobs),wall_seconds=time.perf_counter()-start));print(f'Replayed {len(results)}/{len(jobs)}',flush=True)
 write(RUN/'replay_execution.json',dict(trajectories=len(results),queries=len(results)*32,wall_seconds=time.perf_counter()-start,cpu_seconds=sum(z['cpu_seconds'] for z in results),fresh_objective_fits=0))
if __name__=='__main__':main()
