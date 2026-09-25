"""Finish this authorized run after the already-running collector exits."""
from common import *
import subprocess

def main():
 pid=read(RUN/'launch.json')['pid']
 while not (RUN/'collection.json').exists():
  try:os.kill(pid,0)
  except ProcessLookupError:raise RuntimeError('Collector exited without completion metadata')
  if (RUN/'STOP').exists():raise RuntimeError('STOP requested')
  time.sleep(5)
 for name in ['replay.py','report.py','export.py']:
  print('Starting',name,flush=True)
  subprocess.run([sys.executable,str(E/name)],cwd=ROOT,check=True)
 print('Study analysis and export finished',flush=True)
if __name__=='__main__':main()
