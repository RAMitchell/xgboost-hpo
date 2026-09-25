from common import *
from baselines import PoolBaseline
from scipy.optimize import approx_fprime

def main():
 configs=read(E/'configurations.json');X=encode(configs);rng=np.random.default_rng(773);xx=X[:8];d=(xx[:,None]-xx[None,:])**2;Y=rng.normal(size=(8,12));groups=[(d,Y@Y.T/12,1.)];theta=np.log([.8]*18+[1.,.03]);v,g=objective(theta,groups);numeric=approx_fprime(theta,lambda t:objective(t,groups)[0],1e-6)
 assert np.allclose(g,numeric,atol=2e-5,rtol=2e-4),np.max(np.abs(g-numeric))
 a=dict(configs[0]);a['unit']=[.777]*13;assert np.array_equal(encode([a]),X[:1])
 truth=np.sin(np.arange(len(X)))+np.arange(len(X))*.001
 for method in ['smac','tpe','gp_ei_online','random']:
  opt=PoolBaseline(method,X,72)
  for i in range(10):opt.tell(i,float(truth[i]))
  for _ in range(3):cid=opt.ask();assert cid not in opt.seen;opt.tell(cid,float(truth[cid]))
 with np.load(E/'prior/candidate_prior.npz') as a:
  gp=QuantileGP(a['mean'],a['covariance'],a['noise_variance'],seed=5)
  for _ in range(32):i=gp.ask();gp.tell(i,float(truth[i]))
 assert len(set(gp.seen))==32
 source={s['family'] for s in read(SOURCE/'datasets.json')};targets={s['family'] for s in read(E/'candidates.json')};assert not source&targets
 assert not {key(c['params']) for c in configs}&{key(c['params']) for c in read(SOURCE/'configurations.json')}
 write(E/'checks.json',dict(kernel_gradient_max_error=float(np.max(np.abs(g-numeric))),effective_encoding_invariant=True,all_baseline_synthetic_checks_pass=True,source_target_families_disjoint=True,source_target_configurations_disjoint=True,target_outcomes_read=0))
 print('Source-only and synthetic checks passed',flush=True)
if __name__=='__main__':main()
