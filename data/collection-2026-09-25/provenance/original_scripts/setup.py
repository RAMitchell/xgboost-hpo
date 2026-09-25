from collector import *
import subprocess,importlib.util
assert not (R/'freeze.json').exists()
rows=read(S/'dataset_breadth/source_rows.json');unique={r['did']:r for r in rows};reserved=sorted({r['family'] for r in read(S/'dataset_breadth/evaluation_rows.json')})
# Use stable metadata hashes; no historical outcome is read.
bytype={t:sorted([dict(did=r['did'],dataset=r['dataset'],family=r['family'],problem_type=t) for r in unique.values() if r['problem_type']==t],key=lambda r:key(['collection50-order-v1',r['did']])) for t in ['binary','multiclass','regression']}
order=[]
for i in range(max(map(len,bytype.values()))):
 for t in bytype:
  if i<len(bytype[t]):order.append(bytype[t][i])
assert len(order)==120 and len({r['family'] for r in order})==120 and not set(reserved)&{r['family'] for r in order}
write(R/'candidates.json',order);write(R/'reserved_families.json',reserved)
# Reuse the exact expanded pilot mapping, replacing Sobol with iid uniforms.
source=(S/'database_learning_curve/collector.py').read_text();a=source.index('def design(');b=source.index('\ndef matrices(',a);function=source[a:b].replace('def design(seed, power, prefix):','def design(seed, count, prefix):').replace('qmc.Sobol(d=13,scramble=True,seed=seed).random_base2(power)','np.random.default_rng(seed).random((count,13))').replace("group='sobol'","group='shared_iid'")
ns=dict(np=np,logspace=logspace,spike=spike);exec(function,ns);config=ns['design'](20260926,40,'random');assert len({key(c['params']) for c in config})==40
assert all(c['params']['max_depth']>0 or c['params']['max_leaves']>0 for c in config);write(R/'configurations.json',config);(R/'configuration_mapping.py').write_text('from collector import np,logspace,spike\n'+function)
allowed=os.sched_getaffinity(0);cores={}
for line in subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True).splitlines():
 if line.startswith('#'):continue
 cpu,core,socket=map(int,line.split(','))
 if cpu in allowed:cores.setdefault((socket,core),cpu)
cpus=[v for k,v in sorted(cores.items())][:16];assert len(cpus)==16
write(R/'hardware.json',dict(cpus=cpus,workers=16,threads_per_fit=1,cpu_model=platform.processor(),lscpu=subprocess.check_output(['lscpu'],text=True)))
write(R/'environment.json',dict(python=sys.executable,xgboost=xgb.__version__,library=xgb.core._LIB._name,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__))
paths=list(R.glob('*.py'))+list(R.glob('*.json'))+[R/'PROTOCOL.md',S/'dataset_breadth/source_rows.json',S/'dataset_breadth/evaluation_rows.json',S/'pilot/family_registry.csv',Path(xgb.core._LIB._name)]
hashes={str(p):digest(p) for p in paths};identity=key(hashes);write(R/'freeze.json',dict(identity=identity,hashes=hashes));print(json.dumps(dict(identity=identity,workers=16,cpus=cpus,configs=len(config),candidate_families=len(order))),flush=True)
