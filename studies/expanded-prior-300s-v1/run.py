"""Restart the 68 censored fits, audit prefixes, replay policies and export a delta."""
from common import *
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
import subprocess, traceback, fcntl, gzip, zipfile

OLD=ROOT/'runs/expanded-prior-v1'

def worker(job):
 import collector
 collector.FIT_CPU_CAP=300
 sp,c,identity=job
 try:
  z=collector.evaluate(sp,c,identity)
 except Exception as exc:
  z=dict(record_id=sp['uid']+'__'+c['cid'],uid=sp['uid'],cid=c['cid'],identity=identity,status='failed',params_sampled=c['params'],error=repr(exc),traceback=traceback.format_exc(),total_cpu_seconds=0.)
  write(RUN/'records'/(z['record_id']+'.json'),z)
 return z

def prefix_audit(jobs):
 checks=[]
 for sp,c,_ in jobs:
  rid=sp['uid']+'__'+c['cid'];old=read(OLD/'records'/(rid+'.json'));new=read(RUN/'records'/(rid+'.json'))
  assert new['status']!='failed',(rid,new)
  assert old['params_effective']==new['params_effective']
  with np.load(OLD/'records'/(rid+'.npz')) as a,np.load(RUN/'records'/(rid+'.npz')) as b:
   n=min(len(a['validation_loss']),len(b['validation_loss']))
   delta=max(float(np.max(np.abs(a[k][:n]-b[k][:n]))) for k in ['train_loss','validation_loss'])
   assert all(np.allclose(a[k][:n],b[k][:n],rtol=1e-7,atol=1e-9) for k in ['train_loss','validation_loss']),(rid,delta)
   assert len(b['validation_loss'])>=len(a['validation_loss']),(rid,'shorter curve')
  checks.append(dict(record_id=rid,dataset=sp['dataset'],old_status=old['status'],new_status=new['status'],old_rounds=old['trained_rounds'],new_rounds=new['trained_rounds'],old_loss=old['validation_loss'],new_loss=new['validation_loss'],shared_prefix_rounds=n,max_absolute_difference=delta,train_cpu_seconds=new['train_cpu_seconds']))
 write(E/'reports/prefix_audit.json',checks)
 return checks

def export_delta(jobs):
 import pandas as pd
 destination=ROOT/'data/heldout-expanded-300s-v1';destination.mkdir(exist_ok=True);rows=[];raw=[]
 with zipfile.ZipFile(destination/'replacement_curves.zip','w',compression=zipfile.ZIP_STORED) as archive:
  for sp,c,_ in jobs:
   rid=sp['uid']+'__'+c['cid'];z=read(RUN/'records'/(rid+'.json'));raw.append(z)
   row={k:v for k,v in z.items() if not isinstance(v,(dict,list))};row.update({'param_'+k:v for k,v in z['params_sampled'].items()});rows.append(row)
   p=RUN/'records'/(rid+'.npz')
   if p.exists():archive.write(p,p.name)
 for path,records in [(destination/'replacement_records.jsonl.gz',raw),(E/'reports/trajectories.jsonl.gz',[read(p) for p in sorted((RUN/'trajectories').glob('*.json'))])]:
  with path.open('wb') as f:
   with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0,compresslevel=9) as g:
    for z in records:g.write((json.dumps(z,separators=(',',':'),allow_nan=False)+'\n').encode())
 pd.DataFrame(rows).to_parquet(destination/'replacement_evaluations.parquet',index=False,compression='zstd',compression_level=10)
 write(destination/'collection.json',read(RUN/'collection.json'))
 write(destination/'manifest.json',dict(parent_snapshot='data/heldout-expanded-v1',parent_manifest_sha256=digest(ROOT/'data/heldout-expanded-v1/manifest.json'),replacement_records=len(rows),sha256={p.name:digest(p) for p in sorted(destination.iterdir()) if p.is_file() and p.name!='manifest.json'}))
 write(E/'reports/export.json',dict(replacement_records=len(rows),bytes=sum(p.stat().st_size for p in destination.iterdir()),parent_snapshot='data/heldout-expanded-v1'))


def main():
 f=verify_freeze();(E/'reports').mkdir(exist_ok=True);lock=(RUN/'.execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,h in read(E/'input_manifest.json')['sha256'].items():assert digest(ROOT/name)==h,name
 cpus=read(E/'design.json')['cpus'];os.sched_setaffinity(0,set(cpus));configs=read(E/'configurations.json');datasets=read(RUN/'datasets.json');jobs=[];reused=[]
 for sp in datasets:
  for c in configs:
   old=read(OLD/'records'/(sp['uid']+'__'+c['cid']+'.json'))
   if old['status']=='censored':jobs.append((sp,c,key([f['identity'],sp['data_identity']])))
   else:
    assert old['status']=='complete';assert read(RUN/'records'/(old['record_id']+'.json'))==old;reused.append(old)
 assert len(jobs)==68 and len(reused)==2812
 if not (RUN/'collection.json').exists():
  start=time.perf_counter();results=[];last=0.
  with ProcessPoolExecutor(len(cpus),mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(cpus,)) as pool:
   active={pool.submit(worker,j) for j in jobs}
   while active:
    done,active=wait(active,timeout=5,return_when=FIRST_COMPLETED)
    for future in done:results.append(future.result())
    if time.perf_counter()-last>10 or not active:
     z=dict(state='collecting',completed=len(results),target=len(jobs),reused=len(reused),counts={s:sum(z['status']==s for z in results) for s in ['complete','censored','failed']},elapsed_seconds=time.perf_counter()-start,job_cpu_seconds=sum(z['total_cpu_seconds'] for z in results),workers=len(cpus))
     write(RUN/'progress.json',z);print(json.dumps(z),flush=True);last=time.perf_counter()
  combined=reused+results
  write(RUN/'collection.json',dict(identity=f['identity'],parent_identity=read(ROOT/'studies/expanded-prior-v1/freeze.json')['identity'],datasets=len(datasets),attempted=len(combined),reused=len(reused),fresh_attempted=len(results),counts={s:sum(z['status']==s for z in combined) for s in ['complete','censored','failed']},fresh_counts={s:sum(z['status']==s for z in results) for s in ['complete','censored','failed']},round_cap=sum(z.get('stop_reason')=='round_cap' for z in combined),wall_seconds=time.perf_counter()-start,job_cpu_seconds=sum(z['total_cpu_seconds'] for z in results),training_cpu_seconds=sum(z.get('train_cpu_seconds',0) for z in results),workers=len(cpus),fit_cpu_cap=300))
 checks=prefix_audit(jobs);print('Prefix checks passed for',len(checks),'fits',flush=True)
 for name in ['replay.py','report.py']:
  print('Starting',name,flush=True);subprocess.run([sys.executable,str(E/name)],cwd=ROOT,check=True)
 export_delta(jobs)
 write(RUN/'finished.json',dict(state='complete',collection=read(RUN/'collection.json'),reports=str(E/'reports')))
 print('300-second evaluation finished',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  write(RUN/'error.json',dict(traceback=traceback.format_exc()));raise
