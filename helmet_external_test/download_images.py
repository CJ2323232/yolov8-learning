#!/usr/bin/env python3
"""Download a set of Pexels photos for qualitative YOLO helmet inference."""
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import csv
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'external_test'


# Photo ids and intended challenge cases; check actual images after download.
PHOTOS = [
    ('01_yellow_helmet_closeup', 8960993, '近景黄色安全帽'),
    ('02_worker_hardhat', 30411827, '单人施工场景'),
    ('03_two_workers', 8961623, '多人施工场景'),
    ('04_workers_rebar', 10202856, '钢筋背景多人'),
    ('05_scaffolding_workers', 13963338, '脚手架'),
    ('06_scaffolding_occlusion', 13795569, '复杂脚手架与遮挡'),
    ('07_worker_obstruction', 28663713, '局部遮挡'),
    ('08_worker_walking', 16368435, '远距离行走'),
    ('09_wood_construction', 8961526, '木结构施工'),
    ('10_indoor_construction', 5493673, '室内施工'),
    ('11_woman_hardhat', 8487409, '人物姿态变化'),
    ('12_workers_sky', 32467388, '天空背景'),
    ('13_black_white', 10555726, '黑白场景'),
    ('14_held_helmet', 8487720, '手持安全帽'),
    ('15_helmet_without_wearer', 30592246, '独立安全帽'),
]

# Pexels provides reduced-size variants suitable for model inference.
def candidates(pid):
    stem = f'https://images.pexels.com/photos/{pid}/pexels-photo-{pid}.jpeg'
    return [stem + '?auto=compress&cs=tinysrgb&w=1280', stem]


def is_jpeg(blob):
    return blob.startswith(b'\xff\xd8\xff')


def main():
    OUT.mkdir(exist_ok=True)
    good, failed = 0, []
    for idx, (name, pid, desc) in enumerate(PHOTOS, 1):
        target = OUT / (name + '.jpg')
        if target.exists() and is_jpeg(target.read_bytes()[:8]):
            print(f'[{idx:02d}/15] 已存在: {target.name}')
            good += 1
            continue
        error = None
        for url in candidates(pid):
            try:
                req = Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; personal-educational-research/1.0)',
                    'Accept': 'image/avif,image/webp,image/jpeg,image/*;q=0.8,*/*;q=0.5',
                })
                with urlopen(req, timeout=30) as resp:
                    blob = resp.read(15 * 1024 * 1024 + 1)
                if len(blob) > 15 * 1024 * 1024:
                    raise ValueError('图片超过 15 MB')
                if not is_jpeg(blob):
                    raise ValueError('服务器未返回 JPEG 图片')
                target.write_bytes(blob)
                print(f'[{idx:02d}/15] 下载成功: {target.name} ({len(blob)//1024} KB)')
                good += 1
                error = None
                break
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                error = str(exc)
        if error:
            failed.append((name, pid, error))
            print(f'[{idx:02d}/15] 失败: {name}，原因：{error}')
        time.sleep(0.3)
    with (ROOT / 'photo_sources.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'pexels_photo_id', 'test_scenario', 'source_url'])
        writer.writerows((f'{name}.jpg', pid, desc, f'https://www.pexels.com/photo/{pid}/') for name, pid, desc in PHOTOS)
    print(f'\n已获取 {good}/15 张，文件夹：{OUT}')
    if failed:
        print('以下图片下载失败，可使用 photo_sources.csv 中的来源页面手动下载：')
        for name, pid, err in failed:
            print(f'  {name}.jpg → https://www.pexels.com/photo/{pid}/ ({err})')
    print('图片下载成功后，即可让 YOLO 以 external_test 文件夹为 source 批量预测。')


if __name__ == '__main__':
    main()
