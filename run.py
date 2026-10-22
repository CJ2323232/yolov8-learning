"""YOLO entry point with dataset verification and model/data fingerprints."""
import argparse
import json
import platform
from pathlib import Path

from workflow import RELEASE_SHA256, TRAINING_RECORD, register_training, sha256, verify_dataset, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['check', 'predict', 'train', 'val'])
    p.add_argument('--model', default='yolov8n.pt')
    p.add_argument('--pretrained', type=Path, help='项目标准/P2配置的官方初始化权重；两版只加载共有层0..15')
    p.add_argument('--source', default='https://ultralytics.com/images/bus.jpg')
    p.add_argument('--data', default='coco8.yaml')
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--name')
    p.add_argument('--project', default='runs')
    p.add_argument('--conf', type=float, help='Prediction threshold only; AP evaluation uses 0.001')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--optimizer', choices=['AdamW', 'SGD'], default='AdamW')
    p.add_argument('--lr0', type=float, default=0.001667)
    p.add_argument('--lrf', type=float, default=0.01)
    p.add_argument('--weight-decay', type=float, default=0.0005)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--warmup-epochs', type=float, default=3)
    p.add_argument('--warmup-bias-lr', type=float, default=0.0)
    p.add_argument('--mosaic', type=float, default=1.0)
    p.add_argument('--close-mosaic', type=int, default=10)
    p.add_argument('--scale', type=float, default=0.5)
    p.add_argument('--translate', type=float, default=0.1)
    p.add_argument('--mixup', type=float, default=0.0)
    p.add_argument('--nbs', type=int, default=64)
    p.add_argument('--dry-run', action='store_true', help='验证数据并打印参数，不加载模型、不训练、不运行验证')
    p.add_argument('--resume', action='store_true', help='Resume the explicitly supplied last.pt')
    p.add_argument('--allow-legacy-data', action='store_true', help='Explicit historical diagnosis; never a clean baseline')
    p.add_argument('--training-record', type=Path, default=TRAINING_RECORD)
    return p


def select_device(requested, torch):
    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        return '0'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(args.epochs, args.imgsz, args.batch) < 1:
        p.error('epochs、imgsz、batch 必须为正整数')
    if args.conf is not None and not 0 <= args.conf <= 1:
        p.error('conf 必须在 0 到 1 之间')
    if args.lr0 <= 0 or not 0 < args.lrf <= 1 or args.weight_decay < 0 or args.warmup_epochs < 0 or not 0 <= args.mosaic <= 1 or args.close_mosaic < 0 or not 0 < args.momentum < 1 or args.warmup_bias_lr < 0:
        p.error('训练参数超出范围')
    if args.resume and (args.mode != 'train' or not Path(args.model).is_file()):
        p.error('恢复训练需要 train 模式及明确的 last.pt 路径')
    if not all(0 <= x <= 1 for x in (args.scale, args.translate, args.mixup)) or args.nbs < 1:
        p.error('scale、translate、mixup 应在 0 到 1 之间，nbs 必须为正整数')
    if args.dry_run and args.mode not in ('train', 'val'):
        p.error('--dry-run 仅支持 train 或 val')
    from helmet_models import VARIANTS
    if (args.mode == 'train' and not args.resume and not args.pretrained
            and Path(args.model).resolve() in {path.resolve() for path in VARIANTS.values()}):
        p.error('项目 standard/P2 配置需要明确的预训练初始化；请使用 train_helmet.py --variant standard 或 p2')
    initialization = None
    if args.pretrained:
        if args.mode != 'train' or args.resume:
            p.error('--pretrained 只用于新训练；恢复训练应直接指定 last.pt，不重新初始化')
        from helmet_models import initialization_spec
        initialization = initialization_spec(args.model, args.pretrained)
    identity = verify_dataset(args.data, allow_legacy=args.allow_legacy_data) if args.mode in ('train', 'val') else None
    input_dataset_identity = identity
    model_path = Path(args.model)
    model_hash = sha256(model_path) if model_path.is_file() else None
    if identity and identity.get('manifest_sha256') and model_hash == RELEASE_SHA256:
        if not args.allow_legacy_data:
            p.error('历史公开模型见过旧划分数据，不能用于新划分基准；请从官方预训练权重新训练。')
        identity = {**identity, 'status': 'legacy_diagnostic_not_independent_baseline'}
    if args.dry_run:
        plan = {'status': 'preflight_only_no_training_or_prediction', 'arguments': vars(args),
                'dataset': identity, 'model_sha256': None if initialization else model_hash,
                'initialization': initialization,
                'planned_evaluation': {'split': args.split, 'AP_conf': 0.001,
                    'nms_iou': 0.7, 'max_det': 300},
                'limitations': 'No device memory or training speed measured. Dataset risk retrieval is not exhaustive.'}
        print(json.dumps(plan, ensure_ascii=False, indent=2, default=str))
        return plan
    import torch
    import ultralytics
    from ultralytics import YOLO
    device = select_device(args.device, torch)
    environment = {'python': platform.python_version(), 'system': platform.system(),
                   'machine': platform.machine(), 'torch': torch.__version__,
                   'ultralytics': ultralytics.__version__, 'selected_device': device,
                   'cuda_available': torch.cuda.is_available(), 'mps_available': torch.backends.mps.is_available()}
    print(json.dumps(environment, ensure_ascii=False, indent=2))
    if args.mode == 'check':
        return
    if initialization:
        from helmet_models import build_initialized_model
        model, initialization = build_initialized_model(args.model, args.pretrained, args.seed)
    else:
        model = YOLO(args.model)
    loaded = Path(getattr(model, 'ckpt_path', '') or args.model)
    model_hash = sha256(loaded) if loaded.is_file() else model_hash
    def input_identity():
        return {'model_sha256': None if initialization else model_hash, 'dataset': identity,
                **({'architecture_sha256': initialization['architecture_sha256'],
                    'initialization': initialization} if initialization else {})}
    options = dict(device=device, imgsz=args.imgsz, project=args.project, name=args.name or args.mode)
    effective = None
    if args.mode == 'predict':
        source = int(args.source) if args.source.isdecimal() else args.source
        for _ in model.predict(source=source, save=True, stream=True, batch=args.batch,
                               **({'conf': args.conf} if args.conf is not None else {}), **options):
            pass
        output = Path(model.predictor.save_dir)
        effective = vars(model.predictor.args)
    elif args.mode == 'train':
        if args.resume:
            prior = loaded.resolve().parent.parent/'input_identity.json'
            if not prior.is_file() or json.loads(prior.read_text())['dataset'] != identity:
                raise ValueError('恢复训练缺少匹配的数据身份记录；不能从未登记的旧模型恢复新基准')
            resumed_data = (model.ckpt.get('train_args') or {}).get('data')
            if not resumed_data or verify_dataset(resumed_data) != identity:
                raise ValueError('checkpoint 中的训练数据与请求划分不同')
        def save_start(trainer):
            directory = Path(trainer.save_dir)
            if initialization:
                from helmet_models import shared_parameter_digest
                if shared_parameter_digest(trainer.model) != initialization['shared_parameters_sha256']:
                    raise ValueError('训练器重建后共有预训练参数发生变化，停止训练')
                write_json(directory/'initialization.json', {**initialization,
                    'trainer_shared_parameters_verified_before_first_step': True})
            if not args.resume:
                write_json(directory/'input_identity.json', input_identity())
            write_json(directory/'command.json', vars(args))
            write_json(directory/'environment.json', environment)
            write_json(directory/'effective_args.json', vars(trainer.args))
        model.add_callback('on_train_start', save_start)
        model.train(data=args.data, epochs=args.epochs, batch=args.batch, seed=args.seed,
                    workers=0, optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf,
                    weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
                    momentum=args.momentum, warmup_bias_lr=args.warmup_bias_lr,
                    mosaic=args.mosaic, close_mosaic=args.close_mosaic, deterministic=True,
                    scale=args.scale, translate=args.translate, mixup=args.mixup, nbs=args.nbs,
                    **({'cls_remap': False, 'pretrained': True} if initialization else {}),
                    cos_lr=False, resume=args.resume, **options)
        output = Path(model.trainer.save_dir)
        effective = vars(model.trainer.args)
        # No reference is updated until training produced a real best checkpoint.
        best = output/'weights/best.pt'
        if not best.is_file():
            raise FileNotFoundError(f'训练未生成 best.pt，不登记为完成：{output}')
        if verify_dataset(args.data, allow_legacy=args.allow_legacy_data) != input_dataset_identity:
            raise ValueError('训练过程中数据身份发生变化，拒绝登记产物')
        register_training(best, args.data, identity, environment, effective, args.training_record)
    else:
        validation_output = {}
        def capture(validator):
            validation_output.update(save_dir=str(validator.save_dir), effective=vars(validator.args))
        model.add_callback('on_val_end', capture)
        metrics = model.val(data=args.data, batch=args.batch, workers=0, split=args.split,
                            conf=0.001, iou=0.7, max_det=300, rect=True, augment=False,
                            seed=args.seed, deterministic=True, **options)
        output = Path(validation_output['save_dir'])
        effective = validation_output['effective']
        summary = {key: float(value) for key, value in metrics.results_dict.items()}
        summary.update(model=str(loaded.resolve()), model_sha256=model_hash, dataset=identity,
                       data=args.data, split=args.split, device=device, imgsz=args.imgsz,
                       batch=args.batch, precision_recall_definition='class means at library max-mean-F1 confidence; NOT fixed conf=0.25',
                       ap_definition='class-mean AP at matching IoU 0.50 and 0.50:0.95; NMS IoU is separate',
                       per_class=[{'id': int(c), 'name': metrics.names[int(c)], 'precision': float(metrics.box.p[i]),
                                   'recall': float(metrics.box.r[i]), 'map50': float(metrics.box.ap50[i]),
                                   'map50_95': float(metrics.box.ap[i])} for i,c in enumerate(metrics.box.ap_class_index)])
        write_json(output/'metrics.json', summary)
    write_json(output/'command.json', vars(args))
    write_json(output/'environment.json', environment)
    write_json(output/'effective_args.json', effective)
    write_json(output/'input_identity.json', input_identity())
    print(f'完成，结果目录：{output.resolve()}')
    return output


if __name__ == '__main__':
    main()
