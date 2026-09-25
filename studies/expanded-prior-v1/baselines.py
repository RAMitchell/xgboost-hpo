"""Fixed-pool baseline acquisition adapters, not unrestricted package benchmarks."""
from common import *
import optuna
from optuna.distributions import FloatDistribution,CategoricalDistribution
from optuna.samplers._tpe.sampler import _split_trials,default_gamma
from ConfigSpace import ConfigurationSpace,Configuration,Float,Categorical
from smac import Scenario,HyperparameterOptimizationFacade as HPO
from smac.runhistory.runhistory import RunHistory
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,Matern,WhiteKernel
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)

class PoolBaseline:
 def __init__(self,method,X,seed_):
  self.method=method;self.X=X;self.rng=np.random.default_rng(seed_);self.seed=seed_;self.seen=[];self.values=[]
  if method=='smac':
   cs=ConfigurationSpace(seed=seed_);cs.add([Categorical(n,[0,1]) if j in BINARY else Float(n,(0.,1.)) for j,n in enumerate(FEATURES)])
   self.cc=[Configuration(cs,values={n:int(x[j]) if j in BINARY else float(x[j]) for j,n in enumerate(FEATURES)}) for x in X]
   scenario=Scenario(cs,deterministic=True,seed=seed_,n_trials=32)
   self.model=HPO.get_model(scenario);self.model._rf_opts['n_jobs']=1;self.acquisition=HPO.get_acquisition_function(scenario);self.encoder=HPO.get_runhistory_encoder(scenario);self.history=RunHistory();self.encoder.runhistory=self.history;self.random_design=HPO.get_random_design(scenario)
  elif method=='tpe':
   self.sampler=optuna.samplers.TPESampler(seed=seed_,multivariate=False,n_startup_trials=10)
   self.study=optuna.create_study(direction='minimize',sampler=self.sampler)
   self.dist={n:CategoricalDistribution([0,1]) if j in BINARY else FloatDistribution(0.,1.) for j,n in enumerate(FEATURES)}
 def tell(self,cid,value):
  assert cid not in self.seen and np.isfinite(value)
  if self.method=='smac':self.history.add(self.cc[cid],cost=float(value),seed=0)
  if self.method=='tpe':self.study.add_trial(optuna.trial.create_trial(value=float(value),params={n:int(self.X[cid,j]) if j in BINARY else float(self.X[cid,j]) for j,n in enumerate(FEATURES)},distributions=self.dist))
  self.seen.append(cid);self.values.append(float(value))
 def ask(self):
  pool=np.setdiff1d(np.arange(len(self.X)),self.seen);n=len(self.seen)
  if self.method=='random':return int(self.rng.choice(pool))
  if self.method=='smac':
   xx,yy=self.encoder.transform();self.model._rf_opts['max_leaf_nodes']=max(2,len(xx));self.model.train(xx,yy)
   random=self.random_design.check(1);self.random_design.next_iteration()
   if random:return int(self.rng.choice(pool))
   eta=float(self.model.predict_marginalized(np.array([self.cc[i].get_array() for i in self.seen]))[0].min());self.acquisition.update(model=self.model,eta=eta,num_data=n);scores=np.asarray(self.acquisition([self.cc[i] for i in pool])).ravel()
  elif self.method=='tpe':
   trials=self.study.get_trials(deepcopy=False);below,above=_split_trials(self.study,trials,default_gamma(n));scores=np.zeros(len(pool))
   for j,(name,d) in enumerate(self.dist.items()):
    good=self.sampler._build_parzen_estimator(self.study,{name:d},below,True);bad=self.sampler._build_parzen_estimator(self.study,{name:d},above,False)
    samples={name:np.array([d.to_internal_repr(self.X[i,j]) for i in pool])};scores+=self.sampler._compute_acquisition_func(samples,good,bad)
  elif self.method=='gp_ei_online':
   y=np.asarray(self.values);std=y.std()
   if std<1e-12:return int(self.rng.choice(pool))
   yy=(y-y.mean())/std
   def optimizer(fun,start,bounds):
    th,v,_=fmin_l_bfgs_b(fun,start,bounds=bounds,maxiter=100);return th,v
   k=ConstantKernel(1.,(1e-3,1e3))*Matern(np.ones(self.X.shape[1]),(.01,100.),nu=2.5)+WhiteKernel(.01,(1e-6,1.))
   gp=GaussianProcessRegressor(kernel=k,alpha=1e-10,optimizer=optimizer,n_restarts_optimizer=0,normalize_y=False,random_state=self.seed)
   with warnings.catch_warnings():warnings.simplefilter('ignore');gp.fit(self.X[self.seen],yy)
   mu,sd=gp.predict(self.X[pool],return_std=True);latent=np.sqrt(np.maximum(0,sd**2-gp.kernel_.k2.noise_level));scores=log_ei(mu,latent,float(yy.min()))
  else:raise ValueError(self.method)
  assert not np.isnan(scores).any() and not np.isposinf(scores).any()
  return int(self.rng.choice(pool[scores==scores.max()]))
