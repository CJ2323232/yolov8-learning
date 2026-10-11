"""获取已公开的安全帽权重并校验 SHA256；不会覆盖不同的已有模型。"""
import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHA256 = '55be821e9eefdd0a69caf4abd3fdb2a22277059e21ec155222102b32cef2e650'
URL = 'https://raw.githubusercontent.com/CJ2323232/yolov8-learning/main/models/helmet_yolov8n.pt'


def verify(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() == SHA256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'runs/helmet_50/weights/best.pt')
    args = p.parse_args()
    target = args.output.expanduser().resolve()
    if target.exists():
        if verify(target):
            print(f'已存在且校验通过：{target}')
            return
        raise SystemExit(f'目标文件已存在且校验值不同：{target}。请指定其他 --output，避免覆盖自己的模型。')
    target.parent.mkdir(parents=True, exist_ok=True)
    bundled = ROOT/'models/helmet_yolov8n.pt'
    fd, name = tempfile.mkstemp(prefix='.helmet-', suffix='.tmp', dir=target.parent)
    os.close(fd)
    temp = Path(name)
    try:
        if bundled.is_file():
            shutil.copyfile(bundled, temp)
        else:
            print('正在下载公开模型权重…')
            request = urllib.request.Request(URL, headers={'User-Agent':'yolov8-learning'})
            with urllib.request.urlopen(request, timeout=60) as response, temp.open('wb') as f:
                shutil.copyfileobj(response, f)
        if not verify(temp):
            raise SystemExit('权重 SHA256 校验失败，请重新获取；未安装此文件。')
        # 仅当目标不存在时安装；并发启动也不会覆盖已有文件。
        with target.open('xb') as f, temp.open('rb') as source:
            shutil.copyfileobj(source, f)
        print(f'权重已准备好，SHA256 校验通过：{target}')
    finally:
        temp.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
