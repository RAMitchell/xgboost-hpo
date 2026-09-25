"""Replay all policies from the public objective table; no new XGBoost fits."""
import argparse,os
from pathlib import Path

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=1);parser.add_argument('--dataset-id',type=int,action='append');parser.add_argument('--output',type=Path,default=Path('runs/replay-published-replacement-v1'));args=parser.parse_args()
 if args.workers<1:parser.error('--workers must be positive')
 os.environ['HPO_RUN_DIR']=str(args.output.resolve())
 from common import ROOT,E,RUN,read,write,np,METHODS,verify_freeze,physical_cores,init_worker
 import pandas as pd
 import multiprocessing as mp
 import time,fcntl
 from concurrent.futures import ProcessPoolExecutor,as_completed
 from replay import task
 verify_freeze();data=ROOT/'data/heldout-expanded-replacement-v1';datasets=read(data/'datasets.json');configs=read(data/'configurations.json')
 parent=ROOT/'data/heldout-expanded-v1';table=pd.read_parquet(parent/'evaluations.parquet').set_index('record_id')
 updates=pd.read_parquet(data/'replacement_evaluations.parquet').set_index('record_id')
 table=table.loc[~table.uid.isin(read(data/'manifest.json')['excluded_uids'])]
 table=pd.concat([table.drop(index=table.index.intersection(updates.index)),updates],axis=0)
 assert not table.index.duplicated().any()
 if args.dataset_id:
  unknown=set(args.dataset_id)-{sp['did'] for sp in datasets}
  if unknown:parser.error(f'Unknown dataset IDs: {unknown}')
  datasets=[sp for sp in datasets if sp['did'] in args.dataset_id]
 RUN.mkdir(parents=True,exist_ok=True);lock=(RUN/'.execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (RUN/'trajectories').mkdir(exist_ok=True);write(RUN/'datasets.json',datasets);truth={}
 for sp in datasets:
  rows=table.loc[[sp['uid']+'__'+c['cid'] for c in configs]];valid=rows.loc[rows.status=='complete','validation_loss'];lo=float(valid.min());hi=float(valid.max());penalty=hi+max(hi-lo,abs(hi)*.01,1e-12)
  values=[float(r.validation_loss) if r.status=='complete' else penalty for r in rows.itertuples()]
  truth[sp['uid']]=dict(objective=values,status=rows.status.tolist(),cpu=rows.total_cpu_seconds.tolist(),minimum=lo,maximum=hi,penalty=penalty)
 write(RUN/'outcomes.json',truth);write(RUN/'collection.json',read(data/'collection.json'))
 jobs=[(sp,m,r) for sp in datasets for m in METHODS for r in range(10)];cpus=[c for c in read(E/'design.json')['cpus'] if c in os.sched_getaffinity(0)][:args.workers];start=time.perf_counter();results=[]
 with ProcessPoolExecutor(len(cpus),mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(cpus,)) as pool:
  for future in as_completed([pool.submit(task,job) for job in jobs]):results.append(future.result())
 write(RUN/'replay_execution.json',dict(trajectories=len(results),queries=len(results)*32,wall_seconds=time.perf_counter()-start,cpu_seconds=sum(z['cpu_seconds'] for z in results),fresh_objective_fits=0))
 # Compare proposals directly rather than overwrite the archived reports.
 import gzip,json
 with gzip.open(E/'reports/trajectories.jsonl.gz','rt') as f:expected={(z['uid'],z['method'],z['rep']):z['ids'] for z in map(json.loads,f)}
 matches=sum(z['ids']==expected[(z['uid'],z['method'],z['rep'])] for z in results)
 write(RUN/'parity.json',dict(trajectories=len(results),exact_proposal_matches=matches))
 print(f'Replayed {len(results)} trajectories; {matches} exactly match archived proposals. No objective fits.')
if __name__=='__main__':main()
