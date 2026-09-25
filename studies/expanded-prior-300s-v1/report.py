"""Predeclared family-level regret/rank analysis and compact public export."""
from common import *
import gzip,zipfile
import pandas as pd
from scipy.stats import rankdata,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
 verify_freeze();datasets=read(RUN/'datasets.json');outcomes=read(RUN/'outcomes.json');N=len(datasets);M=len(METHODS);B=32
 byid={s['uid']:i for i,s in enumerate(datasets)};curves=np.zeros((N,M,10,B));raw=np.zeros_like(curves);cost=np.zeros_like(curves);overhead=np.zeros((N,M,10));noncomplete=np.zeros((N,M,10));paths=[];median_scale=np.zeros(N);denoms=[]
 for i,sp in enumerate(datasets):
  z=outcomes[sp['uid']];valid=np.array(z['objective'])[np.array(z['status'])=='complete'];denoms.append(max(z['maximum']-z['minimum'],1e-12));median_scale[i]=np.median(valid)-z['minimum']
 for p in sorted((RUN/'trajectories').glob('*.json')):
  z=read(p);i=byid[z['uid']];j=METHODS.index(z['method']);r=z['rep'];truth=np.asarray(outcomes[z['uid']]['objective']);idx=np.asarray(z['ids']);assert len(set(idx))==B
  assert np.array_equal(truth[idx],np.asarray(z['values']));best=np.minimum.accumulate(truth[idx]);assert np.array_equal(best,z['best'])
  raw[i,j,r]=best;curves[i,j,r]=(best-outcomes[z['uid']]['minimum'])/denoms[i];cost[i,j,r]=z['objective_cpu'];overhead[i,j,r]=z['cpu_seconds'];noncomplete[i,j,r]=z['noncomplete_queries'];paths.append(z)
 assert len(paths)==N*M*10
 means=curves.mean(axis=2);areas=means[:,:,:16].mean(axis=2);rng=np.random.default_rng(20260928);boot=rng.integers(0,N,size=(10000,N));draws=means[boot].mean(axis=1);lower,upper=np.quantile(draws,[.025,.975],axis=0)
 ranks=rankdata(raw,axis=1,method='average').mean(axis=2);rankdraws=ranks[boot].mean(axis=1);ranklo,rankhi=np.quantile(rankdraws,[.025,.975],axis=0)
 summary=[];comparisons=[];signs=rng.choice([-1.,1.],size=(20000,N));pvalues=[]
 for j,(method,label) in enumerate(zip(METHODS,LABELS)):
  a=areas[:,j];z=dict(method=method,label=label,area1_16=float(a.mean()),area95=np.quantile(a[boot].mean(axis=1),[.025,.975]).tolist(),optimizer_cpu_seconds_mean=float(overhead[:,j].mean()),objective_cpu_at32_mean=float(cost[:,j,:,-1].mean()),noncomplete_queries_mean=float(noncomplete[:,j].mean()))
  z.update({f'regret{b}':float(means[:,j,b-1].mean()) for b in [2,4,8,16,32]});summary.append(z)
  if j:
   delta=areas[:,0]-a;diff=delta.mean();bd=delta[boot].mean(axis=1);relative=100*diff/a.mean() if a.mean()>0 else None
   relboot=100*bd/np.maximum(a[boot].mean(axis=1),1e-12);pv=float((1+np.sum(np.abs((signs*delta).mean(axis=1))>=abs(diff)-1e-15))/(len(signs)+1));pvalues.append(pv)
   comparisons.append(dict(comparator=method,prior_minus_comparator=float(diff),difference95=np.quantile(bd,[.025,.975]).tolist(),relative_percent=relative,relative95=np.quantile(relboot,[.025,.975]).tolist(),paired_signflip_p=pv,prior_wins=int((delta<0).sum()),ties=int((delta==0).sum()),prior_losses=int((delta>0).sum())))
 order=np.argsort(pvalues);adjust=np.empty(len(pvalues));running=0.
 for rank,j in enumerate(order):running=max(running,min(1.,pvalues[j]*(len(pvalues)-rank)));adjust[j]=running
 for z,pv in zip(comparisons,adjust):z['holm_p']=float(pv)
 strata=[]
 for name,mask in [('classification',np.array([s['task']=='classification' for s in datasets])),('regression',np.array([s['task']=='regression' for s in datasets]))]:
  if not mask.any():continue
  for j,m in enumerate(METHODS):strata.append(dict(stratum=name,datasets=int(mask.sum()),method=m,area1_16=float(areas[mask,j].mean()),regret16=float(means[mask,j,15].mean()),regret32=float(means[mask,j,31].mean())))
 clean=np.array([all(v=='complete' for v in outcomes[s['uid']]['status']) for s in datasets]);sensitivity=[]
 for j,m in enumerate(METHODS):
  md=median_scale>1e-12;med=(raw[md,j].mean(axis=1)-np.array([outcomes[s['uid']]['minimum'] for s in datasets])[md,None])/median_scale[md,None]
  sensitivity.append(dict(method=m,complete_pool_datasets=int(clean.sum()),complete_pool_area=float(areas[clean,j].mean()) if clean.any() else None,median_scaled_datasets=int(md.sum()),median_scaled_area=float(med[:,:16].mean()) if md.any() else None))
  if clean.any():
   cb=np.random.default_rng(20260929).integers(0,int(clean.sum()),size=(10000,int(clean.sum())));a=areas[clean,j];delta=areas[clean,0]-a;rel=100*delta[cb].mean(axis=1)/np.maximum(a[cb].mean(axis=1),1e-12)
   sensitivity[-1].update(prior_relative_percent=float(100*delta.mean()/max(a.mean(),1e-12)),prior_relative95=np.quantile(rel,[.025,.975]).tolist())
 # Prior diagnostics are evaluated only after freezing and executing all policies.
 with np.load(E/'prior/candidate_prior.npz') as a:prior_mean=a['mean'];prior_sd=np.sqrt(np.diag(a['covariance'])+a['noise_variance'])
 diagnostics=[]
 for s in datasets:
  z=outcomes[s['uid']];y=np.array(z['objective']);ok=np.array(z['status'])=='complete';scores=normal_rank_scores(y[ok]);m=prior_mean[ok];sd=prior_sd[ok];rho=float(spearmanr(m,y[ok]).statistic) if np.ptp(y[ok])>0 and np.ptp(m)>0 else None
  diagnostics.append(dict(uid=s['uid'],problem_type=s['problem_type'],rank_spearman=rho,normal_score_rmse=float(np.sqrt(np.mean((m-scores)**2))),coverage90=float(np.mean(np.abs(m-scores)<=1.6448536269514722*sd))))
 reports=E/'reports';reports.mkdir(exist_ok=True)
 result=dict(datasets=N,configuration_pool=96,budget=B,optimizer_seeds=10,methods=summary,comparisons=comparisons,strata=strata,sensitivity=sensitivity,collection=read(RUN/'collection.json'),replay=read(RUN/'replay_execution.json'),prior=read(E/'prior/model.json'),pointwise_intervals='10000 dataset-family bootstrap draws after averaging optimizer seeds',limitations=['one validation split and objective seed per target','targets held out from this prior but inspected in previous different studies','finite-pool adaptations of SMAC and TPE','40 distinct source configurations','same locally snapshotted development binary, not a pinned release build','intervals conditional on this trained prior and candidate pool'])
 write(reports/'summary.json',result);write(reports/'prior_diagnostics.json',diagnostics)
 pd.DataFrame(summary).to_csv(reports/'metrics.csv',index=False);pd.DataFrame(comparisons).to_csv(reports/'paired_comparisons.csv',index=False);pd.DataFrame(strata).to_csv(reports/'task_types.csv',index=False);pd.DataFrame(sensitivity).to_csv(reports/'sensitivity.csv',index=False)
 np.savez_compressed(reports/'curves.npz',regret=curves,rank=ranks,mean=means.mean(axis=0),lower=lower,upper=upper,uids=np.array([s['uid'] for s in datasets]),methods=np.array(METHODS),objective_cpu=cost,optimizer_cpu=overhead)
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(14,5.3));colors=['#2166ac','#8c510a','#777777','#1b9e77','#984ea3','#e66101','#999999'];x=np.arange(1,33)
 for j,label in enumerate(LABELS):
  axes[0].plot(x,means[:,j].mean(axis=0),label=label,color=colors[j],lw=2);axes[0].fill_between(x,lower[j],upper[j],color=colors[j],alpha=.12)
  axes[1].plot(x,ranks[:,j].mean(axis=0),label=label,color=colors[j],lw=2);axes[1].fill_between(x,ranklo[j],rankhi[j],color=colors[j],alpha=.12)
 axes[0].set_ylabel('Normalized validation regret');axes[1].set_ylabel('Mean rank (among all 7 policies)')
 for ax in axes:ax.set_xlabel('Number of evaluations');ax.set_xlim(1,32);ax.set_xticks([1,4,8,16,24,32]);ax.grid(alpha=.2)
 fig.suptitle(f'300-second limit: {N} held-out families × 96 fresh candidates\n10 optimizer seeds; pointwise 95% family-bootstrap intervals',fontsize=13)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.015),frameon=False);fig.tight_layout(rect=[0,.12,1,.92]);fig.savefig(reports/'regret_and_rank.png',dpi=170,bbox_inches='tight');fig.savefig(reports/'regret_and_rank.pdf',bbox_inches='tight');plt.close(fig)
 rows=['# Expanded-space historical prior: held-out comparison','',f"{N} target families, 96 fresh candidate configurations, budget32,10optimizer seeds. Source:50families/1,999successful records at40distinct configurations.",'','| Method | Area 1–16 | Regret 4 | Regret 8 | Regret 16 | Regret 32 |','|---|---:|---:|---:|---:|---:|']
 for z in summary:rows.append(f"| {z['label']} | {z['area1_16']:.5f} | {z['regret4']:.5f} | {z['regret8']:.5f} | {z['regret16']:.5f} | {z['regret32']:.5f} |")
 rows+=['','Lower is better. Regret is relative to the completed finite-pool optimum, normalized by its max–min. See sensitivity.csv for median scaling and complete-pool-only results.','', '## Paired comparisons','', 'Negative differences favor the historical GP. Intervals resample dataset families after averaging seeds.','', '| Comparator | Relative area difference (%) | 95% interval (%) | Holm p |','|---|---:|---|---:|']
 for z in comparisons:rows.append(f"| {z['comparator']} | {z['relative_percent']:.1f} | [{z['relative95'][0]:.1f}, {z['relative95'][1]:.1f}] | {z['holm_p']:.4f} |")
 rows+=['','## Compute and scope','',f"Reused {result['collection']['reused']} completed fits and retrained {result['collection']['fresh_attempted']} timed-out fits in {result['collection']['wall_seconds']/60:.1f} minutes using up to32cores; {result['collection']['job_cpu_seconds']/3600:.2f} job CPU-hours. Status counts: {result['collection']['counts']}. Replayed {len(paths)} optimizer trajectories with no additional objective training.",'','This is a held-out-family transfer experiment for this prior, not a wholly untouched research benchmark. Package baselines are finite-pool adapters; startup budgets differ as specified in PROTOCOL.md. All methods use the same legal pool. No target test losses were evaluated.']
 (reports/'RESULTS.md').write_text('\n'.join(rows)+'\n')
 print(json.dumps({k:result[k] for k in ['datasets','collection','replay','methods','comparisons']},indent=2))
if __name__=='__main__':main()
