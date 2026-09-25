"""Verify exported results without XGBoost, SMAC, downloads or model training."""
from pathlib import Path
import json,gzip,hashlib,io,zipfile
import numpy as np
import pandas as pd

E=Path(__file__).resolve().parent
ROOT=E.parents[1]
D=ROOT/'data/heldout-expanded-v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 for name,expected in read(E/'freeze.json')['hashes'].items():assert sha(ROOT/name)==expected,name
 for name,expected in read(D/'manifest.json')['sha256'].items():assert sha(D/name)==expected,name
 for name,expected in read(E/'artifact_manifest.json')['sha256'].items():assert sha(ROOT/name)==expected,name
 source=read(ROOT/'data/collection-2026-09-25/datasets.json');targets=read(D/'datasets.json');assert not {s['family'] for s in source}&{s['family'] for s in targets}
 configs=read(D/'configurations.json');table=pd.read_parquet(D/'evaluations.parquet');assert len(table)==len(targets)*96
 with gzip.open(D/'records.jsonl.gz','rt') as f:records={z['record_id']:z for z in map(json.loads,f)}
 assert set(table.record_id)==set(records)
 for sp in targets:
  with np.load(D/'splits'/(sp['uid']+'.npz'),allow_pickle=False) as a:
   rows=np.concatenate([a[n] for n in ['train','validation','test']]);assert len(rows)==len(np.unique(rows))==sp['n_rows']
  with zipfile.ZipFile(D/'curves'/(sp['uid']+'.zip')) as arc:
   for name in arc.namelist():
    z=records[name[:-4]];raw=arc.read(name);assert hashlib.sha256(raw).hexdigest()==z['curves_sha256']
    with np.load(io.BytesIO(raw),allow_pickle=False) as a:assert np.isclose(a['validation_loss'].min(),z['validation_loss']) and len(a['validation_loss'])==z['trained_rounds']
 truth={}
 for sp in targets:
  zs=[records[sp['uid']+'__'+c['cid']] for c in configs];valid=[z['validation_loss'] for z in zs if z['status']=='complete'];hi=max(valid);lo=min(valid);penalty=hi+max(hi-lo,abs(hi)*.01,1e-12);truth[sp['uid']]=np.array([z['validation_loss'] if z['status']=='complete' else penalty for z in zs])
 seen=set();count=0
 with gzip.open(E/'reports/trajectories.jsonl.gz','rt') as f:
  for z in map(json.loads,f):
   k=(z['uid'],z['method'],z['rep']);assert k not in seen;seen.add(k);ids=z['ids'];assert len(ids)==len(set(ids))==32
   vals=truth[z['uid']][ids];assert np.array_equal(vals,z['values']) and np.array_equal(np.minimum.accumulate(vals),z['best']);count+=1
 assert count==len(targets)*7*10
 print(f'OK: {len(targets)} held-out families, {len(table)} objective records, {count} complete replay paths, frozen prior/policies and artifact checksums.')
if __name__=='__main__':main()
