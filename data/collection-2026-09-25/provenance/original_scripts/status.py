from pathlib import Path
import json,os
R=Path(__file__).resolve().parent
out={}
for name in ['pid','progress','preparation_status','completion','failure']:
 p=R/(name+'.json')
 if p.exists():out[name]=json.loads(p.read_text())
pid=out.get('pid',{}).get('pid')
if pid:
 try:
  out['process']={'alive':True,'affinity':sorted(os.sched_getaffinity(pid))}
  children=Path(f'/proc/{pid}/task/{pid}/children')
  out['process']['children']=[{'pid':int(p),'affinity':sorted(os.sched_getaffinity(int(p)))} for p in children.read_text().split()] if children.exists() else []
 except ProcessLookupError:out['process']={'alive':False}
print(json.dumps(out,indent=2))
