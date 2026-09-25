"""Replace Dilbert, reuse completed fits, collect one new pool and replay policies."""
from common import *
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing as mp
import fcntl,subprocess,traceback,gzip,zipfile,shutil
OLD=ROOT/'runs/expanded-prior-v1'
RERUN=ROOT/'runs/expanded-prior-300s-v1'

def worker(job):
 import collector
 collector.FIT_CPU_CAP=300
 sp,c,identity=job
 try:return collector.evaluate(sp,c,identity)
 except Exception as exc:
  z=dict(record_id=sp['uid']+'__'+c['cid'],uid=sp['uid'],cid=c['cid'],identity=identity,status='failed',params_sampled=c['params'],error=repr(exc),traceback=traceback.format_exc(),total_cpu_seconds=0.)
  write(RUN/'records'/(z['record_id']+'.json'),z);return z

def choose_replacement(identity):
 from prepare_targets import prepare
 if (RUN/'datasets.json').exists():return read(RUN/'datasets.json'),read(RUN/'replacement.json')
 admitted=[];replacement=None;source_families={s['family'] for s in read(SOURCE/'datasets.json')};parents=read(RUN/'parent_datasets.json');used={s['family'] for s in parents}
 for c in read(E/'replacement_candidates.json'):
  assert c['family'] not in used|source_families
  write(RUN/'progress.json',dict(state='preparing',dataset=c['dataset'],workers=32))
  try:
   sp=prepare(c,identity);replacement=sp;admitted.append(dict(**c,status='admitted'));print('Prepared replacement:',sp['dataset'],sp['n_train'],'training rows,',len(sp['features']),'features',flush=True)
  except Exception as exc:
   admitted.append(dict(**c,status='excluded',error=repr(exc)));print('Excluded replacement candidate:',c['dataset'],repr(exc),flush=True)
  write(RUN/'admissions.json',admitted)
  if replacement is not None:break
 datasets=[]
 for sp in parents:
  if sp['did']==41163:
   if replacement is not None:datasets.append(replacement)
  else:datasets.append(sp)
 assert not {s['family'] for s in datasets}&source_families
 write(RUN/'datasets.json',datasets);write(RUN/'replacement.json',replacement);return datasets,replacement

def export_delta(replacement):
 import pandas as pd
 D=ROOT/'data/heldout-expanded-replacement-v1';D.mkdir(exist_ok=True);(D/'splits').mkdir(exist_ok=True)
 configs=read(E/'configurations.json');datasets=read(RUN/'datasets.json');ids=['openml_12_r0__eval_011']
 if replacement:ids.extend(replacement['uid']+'__'+c['cid'] for c in configs)
 raw=[];rows=[]
 with zipfile.ZipFile(D/'replacement_curves.zip','w',compression=zipfile.ZIP_STORED) as arc:
  for rid in ids:
   z=read(RUN/'records'/(rid+'.json'));raw.append(z);row={k:v for k,v in z.items() if not isinstance(v,(dict,list))};row.update({'param_'+k:v for k,v in z['params_sampled'].items()});rows.append(row)
   cp=RUN/'records'/(rid+'.npz')
   if z['status']!='failed':arc.write(cp,cp.name)
 portable=[]
 for sp in datasets:
  z={k:v for k,v in sp.items() if k not in ['data_dir','data_hashes']};z['original_data_hashes']={str(k).replace(str(RUN),'{RUN_DIR}').replace(str(OLD),'{PARENT_RUN_DIR}'):v for k,v in sp['data_hashes'].items()};portable.append(z)
  if replacement and sp['uid']==replacement['uid']:
   np.savez_compressed(D/'splits'/(sp['uid']+'.npz'),**{n:np.load(Path(sp['data_dir'])/(n+'_rows.npy')) for n in ['train','validation','test']})
 write(D/'datasets.json',portable);shutil.copy2(E/'configurations.json',D/'configurations.json')
 pd.DataFrame(rows).to_parquet(D/'replacement_evaluations.parquet',index=False,compression='zstd',compression_level=10)
 for path,values in [(D/'replacement_records.jsonl.gz',raw),(E/'reports/trajectories.jsonl.gz',[read(p) for p in sorted((RUN/'trajectories').glob('*.json'))])]:
  with path.open('wb') as f:
   with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0,compresslevel=9) as g:
    for z in values:g.write((json.dumps(z,separators=(',',':'),allow_nan=False)+'\n').encode())
 for n in ['collection.json','admissions.json','outcome_audit.json','replay_execution.json']:write(D/n,read(RUN/n))
 write(D/'manifest.json',dict(parent_snapshot='data/heldout-expanded-v1',parent_manifest_sha256=digest(ROOT/'data/heldout-expanded-v1/manifest.json'),excluded_uids=['openml_41163_r0'],replacement_records=len(ids),sha256={str(p.relative_to(D)):digest(p) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='manifest.json'}))
 write(E/'reports/export.json',dict(replacement_records=len(ids),total_records=len(datasets)*len(configs),bytes=sum(p.stat().st_size for p in D.rglob('*') if p.is_file())))

def main():
 f=verify_freeze();(E/'reports').mkdir(exist_ok=True);lock=(RUN/'.execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,h in read(E/'input_manifest.json')['sha256'].items():assert digest(ROOT/name)==h,name
 rid='openml_12_r0__eval_011'
 with np.load(OLD/'records'/(rid+'.npz')) as a,np.load(RERUN/'records'/(rid+'.npz')) as b:
  n=len(a['validation_loss']);assert len(b['validation_loss'])>=n
  for k in ['train_loss','validation_loss']:assert np.array_equal(a[k],b[k][:n])
 cpus=read(E/'design.json')['cpus'];os.sched_setaffinity(0,set(cpus));start=time.perf_counter();datasets,replacement=choose_replacement(f['identity']);configs=read(E/'configurations.json')
 if not (RUN/'collection.json').exists():
  results=[];jobs=[] if replacement is None else [(replacement,c,key([f['identity'],replacement['data_identity']])) for c in configs]
  with ProcessPoolExecutor(len(cpus),mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(cpus,)) as pool:
   active={pool.submit(worker,j) for j in jobs};last=0.
   while active:
    done,active=wait(active,timeout=5,return_when=FIRST_COMPLETED)
    for future in done:results.append(future.result())
    if time.perf_counter()-last>10 or not active:
     z=dict(state='collecting',dataset=replacement['dataset'],completed=len(results),target=len(jobs),reused=2784,counts={s:sum(z['status']==s for z in results) for s in ['complete','censored','failed']},elapsed_seconds=time.perf_counter()-start,job_cpu_seconds=sum(z['total_cpu_seconds'] for z in results),workers=len(cpus))
     write(RUN/'progress.json',z);print(json.dumps(z),flush=True);last=time.perf_counter()
  records=[read(RUN/'records'/(sp['uid']+'__'+c['cid']+'.json')) for sp in datasets for c in configs]
  write(RUN/'collection.json',dict(identity=f['identity'],parent_identity=read(ROOT/'studies/expanded-prior-v1/freeze.json')['identity'],datasets=len(datasets),attempted=len(records),reused=2784,fresh_attempted=len(results),replacement=None if replacement is None else replacement['dataset'],counts={s:sum(z['status']==s for z in records) for s in ['complete','censored','failed']},fresh_counts={s:sum(z['status']==s for z in results) for s in ['complete','censored','failed']},round_cap=sum(z.get('stop_reason')=='round_cap' for z in records),wall_seconds=time.perf_counter()-start,job_cpu_seconds=sum(z['total_cpu_seconds'] for z in results),training_cpu_seconds=sum(z.get('train_cpu_seconds',0) for z in results),workers=len(cpus),fit_cpu_cap=300))
 for name in ['replay.py','report.py']:
  print('Starting',name,flush=True);subprocess.run([sys.executable,str(E/name)],cwd=ROOT,check=True)
 export_delta(replacement);write(RUN/'finished.json',dict(state='complete',collection=read(RUN/'collection.json'),reports=str(E/'reports')));print('Replacement evaluation finished',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  write(RUN/'error.json',dict(traceback=traceback.format_exc()));raise
