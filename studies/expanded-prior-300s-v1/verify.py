"""Verify the frozen policies, public delta, learning curves and replay outcomes."""
from pathlib import Path
import json,gzip,hashlib,zipfile,io
import numpy as np
import pandas as pd
E=Path(__file__).resolve().parent;ROOT=E.parents[1];D=ROOT/'data/heldout-expanded-300s-v1';P=ROOT/'data/heldout-expanded-v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jsonl(p):
 with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]
def main():
 for name,expected in read(E/'freeze.json')['hashes'].items():assert sha(ROOT/name)==expected,name
 manifest=read(D/'manifest.json');assert sha(P/'manifest.json')==manifest['parent_manifest_sha256']
 for root,m in [(P,read(P/'manifest.json')),(D,manifest)]:
  for name,expected in m['sha256'].items():assert sha(root/name)==expected,name
 original={z['record_id']:z for z in jsonl(P/'records.jsonl.gz')};delta={z['record_id']:z for z in jsonl(D/'replacement_records.jsonl.gz')}
 assert len(original)==2880 and len(delta)==68
 assert set(delta)=={rid for rid,z in original.items() if z['status']=='censored'}
 scalar=pd.read_parquet(D/'replacement_evaluations.parquet').set_index('record_id');assert set(scalar.index)==set(delta)
 with zipfile.ZipFile(D/'replacement_curves.zip') as arc:
  assert len(arc.namelist())==len(delta)
  for rid,z in delta.items():
   assert z['params_effective']==original[rid]['params_effective']
   assert z['status'] in ['complete','censored']
   assert scalar.loc[rid,'validation_loss']==z['validation_loss']
   blob=arc.read(rid+'.npz');assert hashlib.sha256(blob).hexdigest()==z['curves_sha256']
   with np.load(io.BytesIO(blob),allow_pickle=False) as b,zipfile.ZipFile(P/'curves'/(z['uid']+'.zip')) as oldarc:
    assert len(b['validation_loss'])==z['trained_rounds'] and float(b['validation_loss'].min())==z['validation_loss']
    with np.load(io.BytesIO(oldarc.read(rid+'.npz')),allow_pickle=False) as a:
     n=len(a['validation_loss']);assert len(b['validation_loss'])>=n
     for k in ['train_loss','validation_loss']:assert np.allclose(a[k],b[k][:n],rtol=1e-7,atol=1e-9)
 records=original|delta;configs=read(E/'configurations.json');datasets=read(P/'datasets.json');truth={};norm={}
 for sp in datasets:
  zs=[records[sp['uid']+'__'+c['cid']] for c in configs];ys=[z['validation_loss'] for z in zs if z['status']=='complete'];lo=min(ys);hi=max(ys);penalty=hi+max(hi-lo,.01*abs(hi),1e-12)
  truth[sp['uid']]=np.array([z['validation_loss'] if z['status']=='complete' else penalty for z in zs]);norm[sp['uid']]=(lo,max(hi-lo,1e-12))
 trajectories=jsonl(E/'reports/trajectories.jsonl.gz');assert len(trajectories)==2100
 with np.load(E/'reports/curves.npz',allow_pickle=False) as a:
  uids=a['uids'].tolist();methods=a['methods'].tolist();seen=set()
  for z in trajectories:
   k=(z['uid'],z['method'],z['rep']);assert k not in seen;seen.add(k);idx=z['ids'];assert len(idx)==len(set(idx))==32
   y=truth[z['uid']][idx];assert np.array_equal(y,z['values']) and np.array_equal(np.minimum.accumulate(y),z['best'])
   lo,denom=norm[z['uid']];regret=(np.minimum.accumulate(y)-lo)/denom
   assert np.array_equal(regret,a['regret'][uids.index(z['uid']),methods.index(z['method']),z['rep']])
  report=read(E/'reports/summary.json')
  for j,m in enumerate(methods):
   row=next(z for z in report['methods'] if z['method']==m);assert np.isclose(a['regret'][:,j,:,:16].mean(),row['area1_16'])
   assert np.isclose(a['regret'][:,j,:,-1].mean(),row['regret32'])
 checks=dict(records=len(records),replacement_fits=len(delta),unchanged_fits=len(original)-len(delta),trajectories=len(trajectories),checks='freeze, parent and delta hashes, configuration equality, prefix parity, curve minima, oracle, trajectories and reported metrics')
 (E/'reports/verification.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':main()
