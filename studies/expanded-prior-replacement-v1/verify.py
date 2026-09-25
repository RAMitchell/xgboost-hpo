"""Verify the public replacement overlay, frozen prior and complete replay curves."""
from pathlib import Path
import json,gzip,hashlib,zipfile,io
import numpy as np
import pandas as pd
E=Path(__file__).resolve().parent;ROOT=E.parents[1];D=ROOT/'data/heldout-expanded-replacement-v1';P=ROOT/'data/heldout-expanded-v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jsonl(p):
 with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]
def main():
 for name,expected in read(E/'freeze.json')['hashes'].items():assert sha(ROOT/name)==expected,name
 manifest=read(D/'manifest.json');assert sha(P/'manifest.json')==manifest['parent_manifest_sha256']
 for root,m in [(P,read(P/'manifest.json')),(D,manifest)]:
  for name,expected in m['sha256'].items():assert sha(root/name)==expected,name
 if (E/'artifact_manifest.json').exists():
  for name,expected in read(E/'artifact_manifest.json')['sha256'].items():assert sha(ROOT/name)==expected,name
 original={z['record_id']:z for z in jsonl(P/'records.jsonl.gz')};delta={z['record_id']:z for z in jsonl(D/'replacement_records.jsonl.gz')}
 records={rid:z for rid,z in original.items() if z['uid'] not in manifest['excluded_uids']}|delta
 datasets=read(D/'datasets.json');configs=read(D/'configurations.json');assert configs==read(E/'configurations.json')
 assert len(records)==len(datasets)*96 and not any(s['did']==41163 for s in datasets)
 source=read(ROOT/'data/collection-2026-09-25/datasets.json');assert not {s['family'] for s in datasets}&{s['family'] for s in source}
 scalar=pd.read_parquet(D/'replacement_evaluations.parquet').set_index('record_id');assert set(scalar.index)==set(delta)
 with zipfile.ZipFile(D/'replacement_curves.zip') as arc:
  assert len(arc.namelist())==sum(z['status']!='failed' for z in delta.values())
  for rid,z in delta.items():
   assert z['params_sampled']==next(c['params'] for c in configs if c['cid']==z['cid'])
   if z['status']=='failed':continue
   assert scalar.loc[rid,'validation_loss']==z['validation_loss']
   blob=arc.read(rid+'.npz');assert hashlib.sha256(blob).hexdigest()==z['curves_sha256']
   with np.load(io.BytesIO(blob),allow_pickle=False) as b:
    assert len(b['validation_loss'])==z['trained_rounds'] and float(b['validation_loss'].min())==z['validation_loss']
    if rid in original:
     assert rid=='openml_12_r0__eval_011' and z['status']=='complete'
     assert z['params_effective']==original[rid]['params_effective']
     with zipfile.ZipFile(P/'curves'/(z['uid']+'.zip')) as oldarc,np.load(io.BytesIO(oldarc.read(rid+'.npz')),allow_pickle=False) as a:
      n=len(a['validation_loss']);assert len(b['validation_loss'])>=n
      for k in ['train_loss','validation_loss']:assert np.array_equal(a[k],b[k][:n])
 truth={};norm={}
 for sp in datasets:
  split=D/'splits'/(sp['uid']+'.npz')
  if not split.exists():split=P/'splits'/(sp['uid']+'.npz')
  with np.load(split,allow_pickle=False) as a:
   ids=np.concatenate([a[k] for k in ['train','validation','test']]);assert len(ids)==len(np.unique(ids))==sp['n_rows']
  zs=[records[sp['uid']+'__'+c['cid']] for c in configs];ys=[z['validation_loss'] for z in zs if z['status']=='complete'];lo=min(ys);hi=max(ys);penalty=hi+max(hi-lo,.01*abs(hi),1e-12)
  truth[sp['uid']]=np.array([z['validation_loss'] if z['status']=='complete' else penalty for z in zs]);norm[sp['uid']]=(lo,max(hi-lo,1e-12))
 trajectories=jsonl(E/'reports/trajectories.jsonl.gz');assert len(trajectories)==len(datasets)*70
 before={(z['uid'],z['method'],z['rep']):z for z in jsonl(ROOT/'studies/expanded-prior-v1/reports/trajectories.jsonl.gz')};changed={z['uid'] for z in delta.values()};unchanged=0
 with np.load(E/'reports/curves.npz',allow_pickle=False) as a:
  uids=a['uids'].tolist();methods=a['methods'].tolist();seen=set()
  for z in trajectories:
   k=(z['uid'],z['method'],z['rep']);assert k not in seen;seen.add(k);idx=z['ids'];assert len(idx)==len(set(idx))==32
   y=truth[z['uid']][idx];assert np.array_equal(y,z['values']) and np.array_equal(np.minimum.accumulate(y),z['best'])
   lo,denom=norm[z['uid']];regret=(np.minimum.accumulate(y)-lo)/denom
   assert np.array_equal(regret,a['regret'][uids.index(z['uid']),methods.index(z['method']),z['rep']])
   if z['uid'] not in changed:assert z['ids']==before[k]['ids'] and z['values']==before[k]['values'];unchanged+=1
  report=read(E/'reports/summary.json')
  for j,m in enumerate(methods):
   row=next(z for z in report['methods'] if z['method']==m);assert np.isclose(a['regret'][:,j,:,:16].mean(),row['area1_16'])
   assert np.isclose(a['regret'][:,j,:,-1].mean(),row['regret32'])
 result=dict(datasets=len(datasets),records=len(records),delta_records=len(delta),trajectories=len(trajectories),unchanged_trajectories_exactly_matched=unchanged,checks='freeze, public hashes, no prior-family overlap, roster exclusion, splits, parameters, mfeat prefix, curve minima, oracle, trajectories and aggregate metrics')
 # Do not modify a finalized checksum manifest on a verification-only invocation.
 if not (E/'artifact_manifest.json').exists():(E/'reports/verification.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()
