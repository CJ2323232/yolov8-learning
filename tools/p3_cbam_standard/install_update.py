"""Add only standard-CBAM files to an existing project; never touch old code/data/weights."""
import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path,PurePosixPath
import shutil,sys

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def safe(root,relative):
 q=PurePosixPath(relative)
 if q.is_absolute() or '..' in q.parts or '\\' in relative or ':' in relative:raise ValueError('Unsafe path '+relative)
 if not relative.startswith('tools/p3_cbam_standard/'):raise ValueError('Only new standard_cbam files allowed')
 target=(root/relative).resolve()
 if not target.is_relative_to(root.resolve()) or not target.is_relative_to((root/'tools/p3_cbam_standard').resolve()):raise ValueError('Path escapes new experiment')
 if target.suffix.lower() in ['.pt','.onnx','.engine']:raise ValueError('No checkpoints in update')
 return target

def install(bundle,root,check=False):
 bundle=Path(bundle).resolve();root=Path(root).resolve();manifest=json.loads((bundle/'update_manifest.json').read_text(encoding='utf-8'))
 if not (root/'train_helmet.py').is_file():raise ValueError('Put update inside the original YOLO project, next to train_helmet.py')
 for relative,expected in manifest['required_existing_files'].items():
  path=root/relative
  if not path.is_file() or sha(path)!=expected:raise ValueError('Existing dependency differs or missing: '+relative+'; no old files will be overwritten')
 changes=[]
 for relative,digest in manifest['files'].items():
  src=safe(bundle/'payload',relative);dest=safe(root,relative)
  if not src.is_file() or sha(src)!=digest:raise ValueError('Corrupt update: '+relative)
  if dest.exists() and not dest.is_file():raise ValueError('Destination is not a file')
  if not dest.exists() or sha(dest)!=digest:changes.append((relative,src,dest))
 if changes and (root/'runs/helmet_standard_p3_cbam_isolated_640_windows').exists():raise ValueError('Standard-CBAM run already exists; refuse code changes that break its resume identity')
 if check:return {'status':'check_only','new_or_updated_files':len(changes),'training_started':False}
 if not changes:return {'status':'already_installed','training_started':False}
 backup=root/'project_work'/('standard_cbam_install_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));backup.mkdir(parents=True)
 for rel,src,dst in changes:
  if dst.exists():
   old=backup/'before'/rel;old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,old)
 for rel,src,dst in changes:
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
  if sha(dst)!=manifest['files'][rel]:raise IOError('Copied file digest mismatch')
 record={'status':'installed','files':len(changes),'training_started':False,'old_core_files_modified':False,'next':'tools/p3_cbam_standard/00_check.bat'}
 (backup/'record.json').write_text(json.dumps(record,indent=2)+'\n');return record
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent);parser.add_argument('--check',action='store_true');a=parser.parse_args()
 try:print(json.dumps(install(Path(__file__).resolve().parent,a.root,a.check),ensure_ascii=False,indent=2))
 except Exception as e:print('INSTALL STOPPED:',e,file=sys.stderr);sys.exit(1)
