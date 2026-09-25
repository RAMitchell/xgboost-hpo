"""Export the new held-out objective database and replay paths in compact form."""
from common import *
import gzip,zipfile,shutil
import pandas as pd


def clean(v):
 if isinstance(v,dict):return {clean(k):clean(x) for k,x in v.items()}
 if isinstance(v,list):return [clean(x) for x in v]
 if isinstance(v,str):return v.replace(str(RUN),'{RUN_DIR}')
 return v

def main():
 verify_freeze();destination=ROOT/'data/heldout-expanded-v1';assert not destination.exists()
 (destination/'curves').mkdir(parents=True);(destination/'splits').mkdir();datasets=read(RUN/'datasets.json');rows=[];raw=[];portable=[]
 for sp in datasets:
  record={k:v for k,v in sp.items() if k not in ['data_dir','data_hashes']};record['original_data_hashes']=clean(sp['data_hashes']);portable.append(record)
  np.savez_compressed(destination/'splits'/(sp['uid']+'.npz'),**{n:np.load(Path(sp['data_dir'])/(n+'_rows.npy')) for n in ['train','validation','test']})
  with zipfile.ZipFile(destination/'curves'/(sp['uid']+'.zip'),'w',compression=zipfile.ZIP_STORED) as archive:
   for p in sorted((RUN/'records').glob(sp['uid']+'__*.json')):
    if p.name.endswith('.failure_details.json'):continue
    z=clean(read(p));raw.append(z);row={k:v for k,v in z.items() if not isinstance(v,(dict,list))};row.update(problem_type=sp['problem_type'],n_train=sp['n_train'],n_features=len(sp['features']));row.update({'param_'+k:v for k,v in z['params_sampled'].items()})
    if 'structure' in z:row.update({'model_'+k:v for k,v in z['structure'].items() if not isinstance(v,list)})
    rows.append(row);cp=p.with_suffix('.npz')
    if cp.exists():archive.write(cp,arcname=cp.name)
 write(destination/'datasets.json',portable);shutil.copy2(E/'configurations.json',destination/'configurations.json');pd.DataFrame(rows).sort_values(['uid','cid']).to_parquet(destination/'evaluations.parquet',index=False,compression='zstd',compression_level=10)
 with (destination/'records.jsonl.gz').open('wb') as output:
  with gzip.GzipFile(filename='',fileobj=output,mode='wb',mtime=0,compresslevel=9) as f:
   for z in sorted(raw,key=lambda z:z['record_id']):f.write((json.dumps(z,separators=(',',':'),allow_nan=False)+'\n').encode())
 trajectories=[read(p) for p in sorted((RUN/'trajectories').glob('*.json'))]
 with (E/'reports/trajectories.jsonl.gz').open('wb') as output:
  with gzip.GzipFile(filename='',fileobj=output,mode='wb',mtime=0,compresslevel=9) as f:
   for z in trajectories:f.write((json.dumps(z,separators=(',',':'),allow_nan=False)+'\n').encode())
 for name in ['collection.json','admissions.json','outcome_audit.json','replay_execution.json']:write(destination/name,clean(read(RUN/name)))
 paths={str(p.relative_to(destination)):digest(p) for p in sorted(destination.rglob('*')) if p.is_file()};write(destination/'manifest.json',dict(schema_version=1,sha256=paths))
 write(E/'reports/export.json',dict(datasets=len(datasets),evaluations=len(rows),curves=sum(z['status']!='failed' for z in raw),trajectories=len(trajectories),bytes=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()),model_binaries_included=False,raw_tables_included=False))
 print('Exported',destination,flush=True)
if __name__=='__main__':main()
