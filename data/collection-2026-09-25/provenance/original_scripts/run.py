from collector import *
from concurrent.futures import wait,FIRST_COMPLETED
from queue import Queue,Empty,Full
from threading import Thread,Event
import datetime

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def verify():
 f=read(R/'freeze.json')
 for p,h in f['hashes'].items():assert digest(p)==h,p
 return f

def init_worker(cpus):
 idx=(mp.current_process()._identity[0]-1)%len(cpus);os.sched_setaffinity(0,{cpus[idx]})
def worker(sp,c):return evaluate(sp,c,sp['data_identity'])

def produce(queue,halt,identity):
 from prepare_data import prepare
 candidates=read(R/'candidates.json');quotas={'binary':18,'multiclass':17,'regression':15};counts={k:0 for k in quotas};admissions=[];datasets=[];deferred=[]
 def offer(c):
  if halt.is_set() or (R/'STOP').exists():return False
  try:
   write(R/'preparation_status.json',dict(state='preparing',dataset=c['dataset'],did=c['did'],accepted=len(datasets),updated=now()))
   sp=prepare(c,identity);datasets.append(sp);counts[c['problem_type']]+=1;admissions.append(dict(did=c['did'],dataset=c['dataset'],family=c['family'],status='admitted',n_rows=sp['n_rows'],n_features=len(sp['features']),problem_type=c['problem_type']))
   write(R/'datasets.json',datasets);write(R/'admissions.json',admissions)
   while not halt.is_set():
    try:queue.put(sp,timeout=1);break
    except Full:continue
   print(f"prepared {len(datasets)}/50: {sp['dataset']} rows={sp['n_rows']} features={len(sp['features'])}",flush=True)
   return True
  except Exception as e:
   admissions.append(dict(did=c['did'],dataset=c['dataset'],family=c['family'],status='excluded',error=repr(e),traceback=traceback.format_exc()));write(R/'admissions.json',admissions);print(f"admission excluded {c['did']} {c['dataset']}: {e!r}",flush=True);return False
 try:
  for c in candidates:
   if halt.is_set() or (R/'STOP').exists():break
   if counts[c['problem_type']]>=quotas[c['problem_type']]:deferred.append(c);continue
   offer(c)
   if len(datasets)==50:break
  if len(datasets)<50 and not halt.is_set():
   for c in deferred:
    offer(c)
    if len(datasets)==50:break
 finally:
  write(R/'preparation_status.json',dict(state='finished' if len(datasets)==50 else 'incomplete',accepted=len(datasets),by_type=counts,updated=now()))
  while not halt.is_set():
   try:queue.put(None,timeout=1);break
   except Full:continue

def audit(identity):
 datasets=read(R/'datasets.json');config=read(R/'configurations.json');records=[];errors=[]
 assert len({s['family'] for s in datasets})==len(datasets)
 reserved=set(read(R/'reserved_families.json'));assert not reserved&{s['family'] for s in datasets}
 for sp in datasets:
  for path,h in sp['data_hashes'].items():assert digest(path)==h,path
  for c in config:
   p=R/'records'/(sp['uid']+'__'+c['cid']+'.json')
   if not p.exists():errors.append(str(p));continue
   z=read(p);records.append(z);assert z['identity']==sp['data_identity'] and z['params_sampled']==c['params']
   if z['status']=='failed':continue
   for ext,field in [('.ubj.zlib','model_sha256'),('.npz','curves_sha256')]:assert digest(p.with_suffix(ext))==z[field]
   with np.load(p.with_suffix('.npz')) as a:
    v=a['validation_loss'];assert len(v)==z['trained_rounds'] and int(np.argmin(v))==z['best_iteration'];assert np.isclose(v.min(),z['validation_loss']);assert np.isfinite(a['validation_prediction']).all()
   assert np.isclose(z['validation_loss'],z['validation_loss_independent'],rtol=2e-5,atol=2e-6)
   if c['params']['max_depth']>0:assert z['structure']['max_realized_depth']<=c['params']['max_depth']
   if c['params']['max_leaves']>0:assert max(z['structure']['leaves_per_tree'])<=c['params']['max_leaves']
 counts={s:sum(z['status']==s for z in records) for s in ['complete','censored','failed']}
 state='complete' if len(datasets)==50 and len(records)==2000 and counts['complete']==2000 else ('complete_with_censoring' if len(datasets)==50 and len(records)==2000 and not counts['failed'] else 'incomplete')
 report=dict(state=state,identity=identity,datasets=len(datasets),records=len(records),missing=len(errors),counts=counts,round_cap_records=sum(z.get('stop_reason')=='round_cap' for z in records),train_cpu_seconds=sum(z.get('train_cpu_seconds',0) for z in records),job_cpu_seconds=sum(z['total_cpu_seconds'] for z in records),max_worker_rss_mib=max([z['worker_peak_rss_mib'] for z in records] or [0]),workers=16,threads_per_fit=1,updated=now())
 write(R/'completion.json',report);write(R/'progress.json',report)
 paths={str(p.relative_to(R)):digest(p) for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'raw_cache' not in p.parts and p.name not in ['run.log','final_manifest.json','pid.json']}
 write(R/'final_manifest.json',dict(identity=identity,artifacts=paths));print(json.dumps(report),flush=True);return report

def main():
 config=read(R/'configurations.json');f=verify();identity=f['identity'];cpus=read(R/'hardware.json')['cpus'];assert len(cpus)==16
 os.sched_setaffinity(0,set(cpus));start=time.perf_counter();(R/'records').mkdir(exist_ok=True)
 with (S/'followup/.execution.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (R/'completion.json').exists():print('Collection already finalized; inspect completion.json before any new work');return
  queue=Queue(maxsize=2);halt=Event();producer=Thread(target=produce,args=(queue,halt,identity),daemon=True);producer.start();producer_done=False;pending=[];inflight={};done=0;counts={'complete':0,'censored':0,'failed':0};cpu=0.;last=0.;admitted_seen=0
  write(R/'progress.json',dict(state='running',pid=os.getpid(),workers=16,cpus=cpus,expected=2000,completed=0,updated=now()))
  try:
   with ProcessPoolExecutor(16,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(cpus,)) as pool:
    while not producer_done or pending or inflight:
     stopped=(R/'STOP').exists() or time.perf_counter()-start>RUN_WALL_CAP
     if stopped:halt.set();pending.clear();producer_done=True
     if not pending and not producer_done:
      try:
       sp=queue.get(timeout=.1)
       if sp is None:producer_done=True
       else:pending=[(sp,c) for c in config];admitted_seen+=1
      except Empty:pass
     while pending and len(inflight)<16 and not stopped:
      sp,c=pending.pop(0);fu=pool.submit(worker,sp,c);inflight[fu]=(sp,c)
     if inflight:
      finished,_=wait(inflight,timeout=.5,return_when=FIRST_COMPLETED)
      for fu in finished:
       sp,c=inflight.pop(fu)
       try:z=fu.result()
       except Exception as e:
        z=dict(identity=sp['data_identity'],record_id=sp['uid']+'__'+c['cid'],uid=sp['uid'],cid=c['cid'],params_sampled=c['params'],status='failed',error=repr(e),traceback=traceback.format_exc(),total_cpu_seconds=0.,total_wall_seconds=0.,worker_peak_rss_mib=0.);write(R/'records'/(z['record_id']+'.json'),z)
       done+=1;counts[z['status']]+=1;cpu+=z['total_cpu_seconds']
       if done%40==0:print(f"records {done}/2000 counts={counts} cpu={cpu:.1f}s elapsed={time.perf_counter()-start:.1f}s",flush=True)
     if time.perf_counter()-last>2:
      write(R/'progress.json',dict(state='stopping' if stopped else 'running',pid=os.getpid(),workers=16,cpus=cpus,expected=2000,completed=done,datasets_dispatched=admitted_seen,inflight=len(inflight),counts=counts,job_cpu_seconds=cpu,elapsed_seconds=time.perf_counter()-start,updated=now()));last=time.perf_counter()
     if stopped and not inflight:break
   halt.set()
   if stopped:write(R/'progress.json',dict(state='paused',reason='STOP or 24h dispatch ceiling',completed=done,counts=counts,updated=now()));return
   verify();report=audit(identity);report['collection_elapsed_seconds']=time.perf_counter()-start;write(R/'execution.json',report)
  finally:halt.set()
if __name__=='__main__':
 try:main()
 except Exception as e:
  write(R/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),updated=now()));print(traceback.format_exc(),flush=True);raise
