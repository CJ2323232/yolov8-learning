"""Windows launcher using existing Python; logs checks separately from formal runs."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]

def choose_python(root):
 explicit=os.environ.get('YOLO_PYTHON')
 if explicit:return explicit
 # The known game-laptop environment may sit above the nested migration project.
 for parent in [root,*list(root.parents)[:5]]:
  for name in ['.venv-yolo','.venv','venv']:
   candidate=parent/name/'Scripts/python.exe'
   if candidate.is_file():return str(candidate)
 return sys.executable

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['check','train','resume','check-resume','evaluate-val']);args=parser.parse_args()
 python=choose_python(ROOT)
 options={'check':['--check'],'train':['--train'],'resume':['--train','--resume'],'check-resume':['--check','--resume']}
 script=ROOT/'tools/p3_cbam_p2'/('evaluate_validation.py' if args.mode=='evaluate-val' else 'train.py')
 command=[python,'-B',str(script),*options.get(args.mode,[])]
 stamp=datetime.now().strftime('%Y%m%d_%H%M%S');logdir=ROOT/'project_work/p3_cbam_p2_training/logs';logdir.mkdir(parents=True,exist_ok=True);log=logdir/f'{args.mode}_{stamp}.log'
 env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'}
 print('Python:',python);print('Mode:',args.mode);print('Log:',log);print('Reuse the previous training environment; no package installs or upgrades.',flush=True)
 with log.open('x',encoding='utf-8') as f:
  f.write('Command: '+repr(command)+'\n')
  proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
  try:
   for line in proc.stdout:f.write(line);f.flush();print(line,end='',flush=True)
   code=proc.wait()
  except KeyboardInterrupt:
   proc.wait();code=130
 run=ROOT/'runs/helmet_p2_p3_cbam_isolated_640_windows'
 if args.mode in ['train','resume'] and run.exists():
  (run/'launcher_logs').mkdir(exist_ok=True);shutil.copy2(log,run/'launcher_logs'/log.name)
 return code
if __name__=='__main__':sys.exit(main())
