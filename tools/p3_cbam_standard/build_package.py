"""Create one additive, hash-verified P3-CBAM Windows package."""
from pathlib import Path
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
GROUP='standard' if HERE.name=='p3_cbam_standard' else 'p2'
WORK=ROOT/'project_work/p3_cbam_packages_2026-10-27'/GROUP
BUNDLE='P3_CBAM_'+GROUP.upper()+'_Windows_Update'
ARCHIVE_NAME='03_标准版_P3_CBAM_迁移包.zip' if GROUP=='standard' else '04_P2版_P3_CBAM_迁移包.zip'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
 bundle=WORK/BUNDLE
 if bundle.exists():raise FileExistsError(bundle)
 payload=bundle/'payload';payload.mkdir(parents=True)
 files=[p for p in HERE.iterdir() if p.is_file() and p.name!='build_package.py']
 entries={}
 for src in sorted(files):
  relative=src.relative_to(ROOT).as_posix();dst=payload/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);entries[relative]=sha(dst)
 parent='models/configs/helmet_yolov8n.yaml' if GROUP=='standard' else 'models/configs/helmet_yolov8n_p2.yaml'
 dependencies=['helmet_cbam.py','helmet_models.py','workflow.py',parent,'models/configs/sources.json','yolov8n.pt','datasets/helmet_scene_isolated_v1/split_manifest.csv']
 manifest={'experiment':GROUP+'_p3_cbam','files':entries,'required_existing_files':{p:sha(ROOT/p) for p in dependencies},'payload_prefix':HERE.relative_to(ROOT).as_posix()+'/','old_files_overwritten':False,'contains_data_or_weights':False}
 (bundle/'update_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 shutil.copy2(HERE/'install_update.py',bundle/'install_update.py')
 bootstrap=r'''@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
if defined YOLO_PYTHON goto selected
py -3.10 --version >nul 2>&1
if errorlevel 1 goto fallback
py -3.10 -B "%~dp0install_update.py"
goto result
:selected
"%YOLO_PYTHON%" -B "%~dp0install_update.py"
goto result
:fallback
python -B "%~dp0install_update.py"
:result
if errorlevel 1 goto failed
echo Installed. Open the ORIGINAL project tools folder and run this experiment's 00_check.bat first.
pause
exit /b 0
:failed
echo Installation stopped. No training started. Keep the error window open.
pause
exit /b 1
'''
 (bundle/'00_apply_update.bat').write_bytes(bootstrap.replace('\n','\r\n').encode())
 folder='tools/'+HERE.name
 title='标准YOLOv8n＋P3-CBAM' if GROUP=='standard' else 'YOLOv8n-P2＋P3-CBAM（不是原来的P2位置CBAM）'
 text=f'''{title} 迁移包\n\n1.将整个{BUNDLE}文件夹放进上次训练的原项目，与train_helmet.py、datasets同级。\n2.双击00_apply_update.bat；只增加本实验文件，不覆盖已有模型或旧三组代码。\n3.进入原项目{folder}，先双击00_check.bat。\n4.检查通过后，双击同目录01_train_50epochs.bat，才会正式训练50轮。\n5.中断时先03_check_resume.bat，再02_resume.bat，总预算仍50轮。\n\n两个新包可以装进同一个原项目，结果目录互不冲突；6GB游戏本按顺序训练两组，不同时启动。\n本包不含数据、权重和Mac环境，不安装或升级依赖。使用原Python3.10.10、torch2.14.0+cu126、Ultralytics8.4.164环境。\n两组统一在P3/stride8、64通道、Detect前接入同一CBAM，注意力单独seed0初始化保持一致。原P2分支CBAM只保留为历史探索，不计入本轮主四组。\n详细说明见{folder}/README.md。\n'''
 (bundle/'先读这个.txt').write_text(text,encoding='utf-8-sig')
 archive=ROOT/'02_实验结果'/ARCHIVE_NAME
 if archive.exists():raise FileExistsError(archive)
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(bundle.rglob('*')):
   if f.is_file():z.write(f,Path(bundle.name)/f.relative_to(bundle))
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for rel,digest in entries.items():assert hashlib.sha256(z.read(f'{BUNDLE}/payload/{rel}')).hexdigest()==digest
  assert b'\x00' not in z.read(f'{BUNDLE}/00_apply_update.bat')
  assert not any(n.lower().endswith(('.pt','.onnx','.engine')) for n in z.namelist())
 record={'group':GROUP,'archive':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,'sha256':sha(archive),'bundle':str(bundle.relative_to(ROOT)),'payload_files':entries,'required_existing_files':manifest['required_existing_files'],'contains_data_or_weights':False,'training_started':False}
 (WORK/'package_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in record.items() if k not in ['payload_files','required_existing_files']},ensure_ascii=False))
if __name__=='__main__':build()
