"""Apply a verified small update to an EXISTING project, backing up changed files.

This file is copied to the update-folder root by the package builder.
No training, package installation, deletion, or dataset/weights overwrites.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative(root, relative):
    posix = PurePosixPath(relative)
    if posix.is_absolute() or '..' in posix.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Invalid update path: {relative}')
    dest = (root/relative).resolve()
    if not dest.is_relative_to(root.resolve()):
        raise ValueError(f'Update path leaves project: {relative}')
    if posix.parts[0] in {'datasets', 'runs', 'runs2', '.venv', 'VOC2028', 'helmet_dataset'} or dest.suffix == '.pt':
        raise ValueError(f'Update may not overwrite data or weights: {relative}')
    return dest


def install(bundle, root, check=False):
    bundle, root = Path(bundle).resolve(), Path(root).resolve()
    if not (root/'train_helmet.py').is_file() or not (root/'datasets/helmet_scene_isolated_v1/split_manifest.csv').is_file():
        raise ValueError('Place this update folder next to the OLD train_helmet.py, inside the existing project.')
    manifest = json.loads((bundle/'update_manifest.json').read_text(encoding='utf-8'))
    if digest(root/'datasets/helmet_scene_isolated_v1/split_manifest.csv') != manifest['dataset_manifest_sha256']:
        raise ValueError('Existing dataset manifest does not match the original P2 experiment.')
    changes=[]
    # Verify every input and destination BEFORE making the first modification.
    for relative, expected in manifest['files'].items():
        src=safe_relative(bundle/'payload',relative)
        dst=safe_relative(root,relative)
        if not src.is_file() or digest(src)!=expected:
            raise ValueError(f'Update file missing or corrupt: {relative}')
        if dst.exists() and not dst.is_file():
            raise ValueError(f'Expected a file: {relative}')
        if not dst.exists() or digest(dst)!=expected:
            changes.append((relative,src,dst))
    if check:
        return {'status':'check_only','files_to_update':len(changes)}
    if not changes:
        return {'status':'already_installed','files_to_update':0}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=root/'project_work'/f'cbam_update_backup_{stamp}'
    backup.mkdir(parents=True,exist_ok=False)
    record={'time':stamp,'files':[]}
    for relative,src,dst in changes:
        item={'path':relative,'previous_sha256':digest(dst) if dst.exists() else None,'new_sha256':digest(src)}
        if dst.exists():
            saved=backup/'before'/relative
            saved.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(dst,saved)
        record['files'].append(item)
    (backup/'update_record.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    for relative,src,dst in changes:
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        if digest(dst)!=manifest['files'][relative]:
            raise IOError(f'Update verification failed: {relative}; backups: {backup}')
    return {'status':'installed','files_to_update':len(changes),'backup':str(backup),
            'next':'tools/windows_cbam/00_check.bat (checks only; no training)'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    print(json.dumps(install(Path(__file__).resolve().parent,args.root,args.check),indent=2,ensure_ascii=False))


if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'UPDATE STOPPED: {exc}',file=sys.stderr)
        sys.exit(1)
