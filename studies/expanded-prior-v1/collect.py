"""Collect the frozen target pool once, with bounded preparation and 16 workers."""
from common import *
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from threading import Thread,Event
from queue import Queue,Empty,Full
import multiprocessing as mp
import fcntl,traceback

def worker(job):
 from collector import evaluate
 sp,c,identity=job
 try:return evaluate(sp,c,identity)
 except Exception as exc:
  z=dict(record_id=sp['uid']+'__'+c['cid'],uid=sp['uid'],cid=c['cid'],identity=identity,status='failed',params_sampled=c['params'],error=repr(exc),traceback=traceback.format_exc(),total_cpu_seconds=0.)
  write(RUN/'records'/(z['record_id']+'.json'),z);return z

def producer(queue,halt,identity):
 from prepare_targets import prepare
 candidates=read(E/'candidates.json');quotas={'binary':12,'multiclass':11,'regression':7};counts={k:0 for k in quotas};datasets=[];admissions=[];deferred=[]
 def offer(c):
  if halt.is_set() or (RUN/'STOP').exists():return
  try:
   write(RUN/'preparation.json',dict(dataset=c['dataset'],admitted=len(datasets)))
   sp=prepare(c,identity);datasets.append(sp);counts[c['problem_type']]+=1
   admissions.append(dict(**c,status='admitted',rows=sp['n_rows'],features=len(sp['features'])))
   write(RUN/'datasets.json',datasets);write(RUN/'admissions.json',admissions)
   while not halt.is_set():
    try:queue.put(sp,timeout=1);break
    except Full:pass
   print(f"Prepared {len(datasets)}/30: {sp['dataset']}, {sp['n_train']} train rows",flush=True)
  except Exception as exc:
   admissions.append(dict(**c,status='excluded',error=repr(exc)));write(RUN/'admissions.json',admissions)
   print(f"Excluded {c['dataset']}: {exc!r}",flush=True)
 try:
  for c in candidates:
   if halt.is_set() or (RUN/'STOP').exists():break
   if counts[c['problem_type']]>=quotas[c['problem_type']]:deferred.append(c);continue
   offer(c)
   if len(datasets)>=30:break
  if len(datasets)<30:
   for c in deferred:
    offer(c)
    if len(datasets)>=30 or halt.is_set():break
 finally:
  while not halt.is_set():
   try:queue.put(None,timeout=1);break
   except Full:pass

def main():
 f=verify_freeze();RUN.mkdir(parents=True,exist_ok=True);(RUN/'records').mkdir(exist_ok=True)
 if (RUN/'collection.json').exists():print('Collection already finalized');return
 cpus=read(E/'design.json')['cpus'];os.sched_setaffinity(0,set(cpus));lock=(RUN/'.execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.perf_counter();config=read(E/'configurations.json');queue=Queue(maxsize=2);halt=Event();thread=Thread(target=producer,args=(queue,halt,f['identity']),daemon=True);thread.start();done=False;pending=[];active={};results=[];last=0.
 try:
  with ProcessPoolExecutor(len(cpus),mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(cpus,)) as pool:
   while not done or pending or active:
    stopped=(RUN/'STOP').exists() or time.perf_counter()-start>86400
    if stopped:halt.set();pending=[];done=True
    if not pending and not done:
     try:
      sp=queue.get(timeout=.1)
      if sp is None:done=True
      else:pending=[(sp,c,key([f['identity'],sp['data_identity']])) for c in config]
     except Empty:pass
    while pending and len(active)<len(cpus):
     job=pending.pop(0);active[pool.submit(worker,job)]=job
    if active:
     complete,_=wait(active,timeout=.5,return_when=FIRST_COMPLETED)
     for future in complete:active.pop(future);results.append(future.result())
    if time.perf_counter()-last>5:
     report=dict(state='collecting',completed=len(results),target=2880,counts={s:sum(z['status']==s for z in results) for s in ['complete','censored','failed']},elapsed_seconds=time.perf_counter()-start,job_cpu_seconds=sum(z['total_cpu_seconds'] for z in results),workers=len(cpus))
     write(RUN/'progress.json',report);last=time.perf_counter()
     print(json.dumps(report),flush=True)
  if stopped:write(RUN/'progress.json',dict(state='paused',completed=len(results)));return
  ds=read(RUN/'datasets.json');assert not {s['family'] for s in ds}&{s['family'] for s in read(SOURCE/'datasets.json')}
  assert len(results)==len(ds)*96 and len({z['record_id'] for z in results})==len(results)
  counts={s:sum(z['status']==s for z in results) for s in ['complete','censored','failed']}
  write(RUN/'collection.json',dict(identity=f['identity'],datasets=len(ds),attempted=len(results),counts=counts,round_cap=sum(z.get('stop_reason')=='round_cap' for z in results),wall_seconds=time.perf_counter()-start,job_cpu_seconds=sum(z['total_cpu_seconds'] for z in results),training_cpu_seconds=sum(z.get('train_cpu_seconds',0) for z in results),workers=len(cpus)))
  print('Target collection finished',flush=True)
 finally:halt.set()
if __name__=='__main__':main()
