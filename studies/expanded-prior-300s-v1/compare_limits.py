"""Summarize changes between 120s and 300s without mixing regret reference scales."""
from pathlib import Path
import json,gzip
import numpy as np
import pandas as pd
E=Path(__file__).resolve().parent;ROOT=E.parents[1];OLD=ROOT/'studies/expanded-prior-v1';D=ROOT/'data/heldout-expanded-v1';DELTA=ROOT/'data/heldout-expanded-300s-v1'
def read(p):return json.loads(p.read_text())
def records(p):
 with gzip.open(p,'rt') as f:return {z['record_id']:z for z in map(json.loads,f)}
def main():
 old=read(OLD/'reports/summary.json');new=read(E/'reports/summary.json');audit=read(E/'reports/prefix_audit.json')
 old_records=records(D/'records.jsonl.gz');replacement=records(DELTA/'replacement_records.jsonl.gz');updated=old_records|replacement
 datasets=read(D/'datasets.json');configs=read(D/'configurations.json');affected=[]
 for sp in datasets:
  ids=[sp['uid']+'__'+c['cid'] for c in configs]
  if not set(ids)&set(replacement):continue
  a=[old_records[rid] for rid in ids];b=[updated[rid] for rid in ids]
  amin=min(z['validation_loss'] for z in a if z['status']=='complete');bmin=min(z['validation_loss'] for z in b if z['status']=='complete')
  best=min((z for z in b if z['status']=='complete'),key=lambda z:z['validation_loss'])
  affected.append(dict(uid=sp['uid'],dataset=sp['dataset'],old_censored=sum(z['status']=='censored' for z in a),new_censored=sum(z['status']=='censored' for z in b),old_best_complete=amin,new_best_complete=bmin,new_best_complete_id=best['cid'],remaining_censored_beating_best=sum(z['status']=='censored' and z['validation_loss']<bmin for z in b)))
 comparisons=[]
 for a in old['methods']:
  b=next(z for z in new['methods'] if z['method']==a['method']);comparisons.append(dict(method=a['method'],old_area=a['area1_16'],new_area=b['area1_16'],old_regret32=a['regret32'],new_regret32=b['regret32']))
 pd.DataFrame(comparisons).to_csv(E/'reports/limit_comparison.csv',index=False)
 pd.DataFrame(audit).to_csv(E/'reports/retrained_configurations.csv',index=False)
 # The 28 unaffected pools must produce exactly the same proposals and observations.
 def paths(p):
  with gzip.open(p,'rt') as f:return {(z['uid'],z['method'],z['rep']):z for z in map(json.loads,f)}
 before=paths(OLD/'reports/trajectories.jsonl.gz');after=paths(E/'reports/trajectories.jsonl.gz');changed={s['uid'] for s in affected};same=0
 for k,z in after.items():
  if k[0] in changed:continue
  assert z['ids']==before[k]['ids'] and z['values']==before[k]['values'];same+=1
 summary=dict(affected_datasets=affected,unaffected_trajectories_exactly_matched=same,retrained=len(audit),rescued=sum(z['new_status']=='complete' for z in audit),still_censored=sum(z['new_status']=='censored' for z in audit),prefix_max_absolute_difference=max(z['max_absolute_difference'] for z in audit),comparison_scale_warning='Each limit has its own completed-pool optimum and denominator. Old/new normalized regret is not a common-scale raw-loss comparison.')
 (E/'reports/limit_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
 rows=['# Interpretation of the five-minute rerun','',f"Retrained {summary['retrained']} fits: {summary['rescued']} now complete, {summary['still_censored']} remain censored. Reused all 2,812 previously completed fits. All {same} trajectories on the 28 unaffected pools match exactly.",'','| Dataset | Timeouts at120s | Timeouts at300s | Best completed loss at120s | Best completed loss at300s |','|---|---:|---:|---:|---:|']
 for z in affected:rows.append(f"| {z['dataset']} | {z['old_censored']} | {z['new_censored']} | {z['old_best_complete']:.6f} | {z['new_best_complete']:.6f} |")
 rows+=['','## Historical-prior comparison at 300 seconds','']
 for z in new['comparisons']:
  rows.append(f"- Versus `{z['comparator']}`: {z['relative_percent']:.1f}% early-area difference (negative favors the prior); 95% family-bootstrap interval [{z['relative95'][0]:.1f}%,{z['relative95'][1]:.1f}%]; Holm-adjusted paired sign-flip p={z['holm_p']:.4f}; prior wins {z['prior_wins']}/30 families.")
 rows+=['','## Limits on interpretation','',summary['comparison_scale_warning'],'','The primary result still treats timeouts as failed queries, even if their saved best-prefix loss is good. This follow-up changes the timeout alone. It does not evaluate a best-model-within-budget objective. Historical training records and the learned prior were not refitted. Target test losses were not used.','', 'Training CPU times exclude preparation/export. The new reruns used 32 cores; reused fits retain their original measured timings. Do not interpret these mixed timings as a newly measured end-to-end 32-core speed benchmark.']
 (E/'reports/INTERPRETATION.md').write_text('\n'.join(rows)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
