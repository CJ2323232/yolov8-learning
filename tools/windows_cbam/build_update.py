"""Package only reviewed code/config/docs. No datasets, checkpoints or venv."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'project_work/p2_cbam_training_2026-10-25'


def build():
    bundle=WORK/'P2_CBAM_Windows_Update'
    payload=bundle/'payload'
    if bundle.exists():
        raise FileExistsError(f'Package staging already exists: {bundle}')
    files=['train_p2_cbam.py','train_helmet.py','helmet_cbam.py','helmet_models.py','workflow.py','run.py',
           'models/configs/helmet_yolov8n.yaml','models/configs/helmet_yolov8n_p2.yaml',
           'models/configs/helmet_yolov8n_p2_cbam.yaml','models/configs/sources.json',
           'models/configs/p2_cbam_sources.json','models/configs/p2_cbam_training_protocol.json',
           'models/configs/p2_reference_parameters.json','docs/P2_CBAM.md','docs/P2_CBAM_TRAINING.md',
           'tools/check_p2_cbam.py','tests/test_p2_cbam.py','tests/test_cbam_training.py',
           'tests/test_cbam_update.py','tests/test_helmet_models.py']
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'tools/windows_cbam').iterdir()
              if p.is_file() and p.name!='build_update.py']
    hashes={}
    for rel in sorted(files):
        source=ROOT/rel
        dest=payload/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
        hashes[rel]=hashlib.sha256(dest.read_bytes()).hexdigest()
    (bundle/'update_manifest.json').write_text(json.dumps({
        'dataset_manifest_sha256':'5781ecdffa173010f539189b15565220f5e2ac597b5f8ea72f5cf7b1e8bf63e4',
        'files':hashes},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    shutil.copyfile(ROOT/'tools/windows_cbam/install_update.py',bundle/'install_update.py')
    bat=r'''@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
if defined YOLO_PYTHON goto explicit
py -3.10 --version >nul 2>&1
if errorlevel 1 goto fallback
py -3.10 -B install_update.py
goto result
:explicit
"%YOLO_PYTHON%" -B install_update.py
goto result
:fallback
python -B install_update.py
:result
if errorlevel 1 goto failed
echo Update installed. Next: open tools\windows_cbam\00_check.bat in the OLD project.
pause
exit /b 0
:failed
echo Update stopped. Keep this window open and share the error. No training was started.
pause
exit /b 1
'''
    (bundle/'00_apply_update.bat').write_bytes(bat.replace('\n','\r\n').encode('utf-8'))
    (bundle/'先读这个.txt').write_text('P2＋CBAM Windows 小更新包\n\n1.把整个P2_CBAM_Windows_Update文件夹放进原项目，与旧train_helmet.py和datasets同级。\n2.双击00_apply_update.bat，自动备份旧代码，再更新代码和配置。\n3.进入原项目tools/windows_cbam，先运行00_check.bat。\n4.检查通过后运行01_train.bat，才会开始50轮训练。\n5.中断后先03_check_resume.bat核查，再02_resume.bat续训。\n\n参数、数据划分、初始化策略与原P2固定一致。不包含图片、模型权重或Mac虚拟环境，不安装/升级依赖。\n沿用原Windows环境：Python3.10.10、torch2.14.0+cu126、Ultralytics8.4.164。\n细节见安装后的docs/P2_CBAM_TRAINING.md。\n',encoding='utf-8-sig')
    archive=ROOT/'02_实验结果/01_P2_CBAM_Windows更新包.zip'
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(bundle.rglob('*')):
            if f.is_file(): z.write(f,Path(bundle.name)/f.relative_to(bundle))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise ValueError('Archive CRC failure')
        for rel,digest in hashes.items():
            if hashlib.sha256(z.read(f'{bundle.name}/payload/{rel}')).hexdigest()!=digest:
                raise ValueError(f'Archive hash mismatch: {rel}')
    record={'archive':str(archive.relative_to(ROOT)), 'bytes':archive.stat().st_size,
            'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'payload_files':hashes,'contains_data_or_weights':False}
    (WORK/'package_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k!='payload_files'},ensure_ascii=False,indent=2))


if __name__=='__main__': build()
