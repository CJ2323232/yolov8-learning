"""在同一验证集上比较输入尺寸，保存指标与库内推理耗时；不训练、不使用测试集调参。"""
import argparse
import csv
import hashlib
import json
import platform
from datetime import date
from pathlib import Path
from workflow import ROOT, RELEASE_SHA256, latest_training, select_model, sha256, verify_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--data')
    parser.add_argument('--allow-legacy-data', action='store_true', help='历史诊断，不作为新基准')
    parser.add_argument('--sizes', type=int, nargs='+', default=[416, 640, 960])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--output', default='reports/size_comparison_new')
    args = parser.parse_args()
    if min([args.batch, *args.sizes]) < 1:
        parser.error('batch 和 sizes 必须为正整数')
    import torch
    import ultralytics
    from ultralytics import YOLO
    from run import select_device
    checkpoint, origin = select_model(args.model, require_training=True)
    record = latest_training() if origin == 'registered_training' else None
    data = args.data or (record['data'] if record else str(ROOT/'helmet_dataset_clean/data.yaml'))
    identity = verify_dataset(data, allow_legacy=args.allow_legacy_data)
    if record and identity.get('manifest_sha256') != record['dataset']['manifest_sha256']:
        parser.error('模型与数据划分的登记身份不一致')
    fingerprint = sha256(checkpoint)
    if fingerprint == RELEASE_SHA256 and identity.get('manifest_sha256'):
        if not args.allow_legacy_data:
            parser.error('历史权重不能用于新划分的独立基准，请先重新训练')
        identity = {**identity, 'status': 'legacy_diagnostic_not_independent_baseline'}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # 避免覆盖上次实验
    device = select_device(args.device, torch)
    model = YOLO(str(checkpoint))
    rows = []
    for size in args.sizes:
        metrics = model.val(data=data, split='val', imgsz=size, batch=args.batch,
                            device=device, workers=0, seed=0, deterministic=True,
                            conf=0.001, iou=0.7, max_det=300, rect=True, augment=False,
                            plots=False, verbose=False, project=str(out/'runs'),
                            name=f'val_{size}')
        rows.append({'imgsz':size, 'precision':float(metrics.box.mp),
                     'recall':float(metrics.box.mr), 'map50':float(metrics.box.map50),
                     'map50_95':float(metrics.box.map),
                     **{f'{k}_ms':float(v) for k,v in metrics.speed.items()}})
        (out/'partial.json').write_text(json.dumps(rows, indent=2))
    report = {'date':str(date.today()), 'experiment':'same checkpoint, validation split, input-size comparison',
              'model_sha256':fingerprint, 'dataset':identity,
              'precision_recall_definition':'class means at max-mean-F1 confidence, not fixed deployment threshold',
              'torch_threads':torch.get_num_threads(),
              'split':'val','device':device,'batch':args.batch,'seed':0,
              'conf':0.001,'iou':0.7,'max_det':300,'rect':True,'augment':False,
              'repeats':1,'order':args.sizes,'python':platform.python_version(),
              'torch':torch.__version__,'ultralytics':ultralytics.__version__,
              'timing_note':'Ultralytics validator inference ms/image; excludes preprocessing, postprocessing and UI. Single pass; not a production throughput benchmark.',
              'results':rows}
    (out/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
