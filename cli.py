"""Portable local-image inference and audited evaluation; historical runners stay intact."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path, PureWindowsPath
import tempfile
import time

from workflow import sha256, verify_dataset

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def probability(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError('must be between 0 and 1')
    return number


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check-data', help='verify one canonical dataset root')
    predict = commands.add_parser('predict', help='predict local images with an existing checkpoint')
    val = commands.add_parser('val', help='delegate AP evaluation to the original run.py')
    for child in (check, val):
        child.add_argument('--data', type=Path, required=True)
        child.add_argument('--data-root', type=Path, help='override YAML path; relative to current directory')
        child.add_argument('--allow-legacy-data', action='store_true', help='explicit unverified-data diagnosis only')
    check.add_argument('--write-config', type=Path, help='save canonical YAML; refuses an existing file')
    for child in (predict, val):
        child.add_argument('--weights', type=Path, required=True, help='existing trusted local .pt; no download')
        child.add_argument('--sha256', help='optional expected checkpoint SHA256')
        child.add_argument('--imgsz', type=positive_int, default=640)
        child.add_argument('--device', default='auto')
        child.add_argument('--output', type=Path, required=True, help='new directory; refuses overwrite')
        child.add_argument('--dry-run', action='store_true', help='check inputs without loading weights')
    predict.add_argument('--source', type=Path, required=True, help='local image or directory (recursive)')
    predict.add_argument('--conf', type=probability, default=0.25)
    predict.add_argument('--iou', type=probability, default=0.7, help='NMS IoU, not matching IoU')
    val.add_argument('--split', choices=('val', 'test'), default='val')
    val.add_argument('--batch', type=positive_int, default=2)
    return p


def existing_file(value, description):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{description} not found: {path}')
    return path


def canonical_data(data, data_root=None):
    """YAML-relative dataset path, or explicit CWD-relative override; never library-global defaults."""
    import yaml
    source = existing_file(data, 'data YAML')
    cfg = yaml.safe_load(source.read_text(encoding='utf-8-sig'))
    if not isinstance(cfg, dict):
        raise ValueError('data YAML must contain a mapping')
    names = cfg.get('names')
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or names != {0: 'hat', 1: 'person'}:
        raise ValueError('helmet class mapping must be 0=hat, 1=person (head without helmet)')
    raw = data_root if data_root is not None else cfg.get('path', '.')
    if not isinstance(raw, (str, Path)):
        raise ValueError('dataset path must be a string')
    if PureWindowsPath(str(raw)).is_absolute() and not Path(str(raw)).is_absolute():
        raise ValueError('foreign Windows path: supply --data-root for this computer')
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() if data_root is not None else source.parent) / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'dataset root not found: {root}; use --data-root if it moved')
    # Keep the audited split convention. Do not carry download scripts into runtime YAML.
    for split in ('train', 'val', 'test'):
        if cfg.get(split) != f'images/{split}':
            raise ValueError(f'{split} must be images/{split} for this audited helmet workflow')
        if not (root / cfg[split]).is_dir():
            raise FileNotFoundError(f'missing {split} images: {root / cfg[split]}')
    normalized = {'path': str(root), 'train': 'images/train', 'val': 'images/val',
                  'test': 'images/test', 'names': names}
    return source, root, normalized


@contextmanager
def verified_data(data, data_root=None, allow_legacy=False):
    import yaml
    source, root, cfg = canonical_data(data, data_root)
    with tempfile.TemporaryDirectory(prefix='helmet-data-') as directory:
        runtime = Path(directory) / 'data.yaml'
        runtime.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        identity = verify_dataset(runtime, allow_legacy=allow_legacy)
        yield runtime, {'source_yaml': str(source), 'dataset_root': str(root),
                        'dataset': identity}, cfg


def image_inputs(source):
    source = Path(source).expanduser().resolve()
    if source.is_file():
        images = [source] if source.suffix.lower() in IMAGE_SUFFIXES else []
    elif source.is_dir():
        images = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise FileNotFoundError(f'image source not found: {source}')
    if not images:
        raise ValueError(f'no supported images in {source}')
    return source, images


def checkpoint(args):
    weights = existing_file(args.weights, 'checkpoint')
    if weights.suffix.lower() != '.pt':
        raise ValueError('this entry accepts local .pt checkpoints only')
    digest = sha256(weights)
    if args.sha256 is not None and digest != args.sha256.lower():
        raise ValueError('checkpoint SHA256 mismatch')
    return weights, digest


def new_output(value):
    output = Path(value).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'output already exists: {output}; choose a new directory')
    return output


def load_model(weights):
    from helmet_cbam import register_cbam
    from ultralytics import YOLO
    register_cbam()
    return YOLO(str(weights), task='detect')


def prediction(args, model_factory=load_model):
    from workflow import write_json
    weights, digest = checkpoint(args)
    source, images = image_inputs(args.source)
    output = new_output(args.output)
    plan = {'mode': 'predict', 'weights': str(weights), 'weights_sha256': digest,
            'source': str(source), 'images': len(images), 'imgsz': args.imgsz,
            'conf': args.conf, 'nms_iou': args.iou, 'device': args.device, 'output': str(output)}
    if args.dry_run:
        return {**plan, 'status': 'inputs_checked_model_not_loaded'}
    import torch
    from PIL import Image
    from run import select_device
    device = select_device(args.device, torch)
    model = model_factory(weights)  # one load for the entire command
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'run.json', {**plan, 'device': device, 'status': 'running'})
    started, records = time.perf_counter(), []
    for image in images:
        results = model.predict(source=str(image), imgsz=args.imgsz, conf=args.conf,
                                iou=args.iou, device=device, batch=1, save=False, verbose=False)
        if len(results) != 1:
            raise ValueError(f'expected one result for {image.name}')
        result = results[0]
        # Include the original extension in the name to avoid a.jpg/a.png collisions.
        relative = image.relative_to(source) if source.is_dir() else Path(image.name)
        rendered = output / 'images' / relative.parent / (relative.name + '.png')
        rendered.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.plot()[..., ::-1]).save(rendered)
        boxes = result.boxes
        detections = [{'class_id': int(cls), 'class_name': str(result.names[int(cls)]),
                       'confidence': float(conf), 'xyxy': [float(v) for v in xyxy]}
                      for cls, conf, xyxy in zip(boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(),
                                                boxes.xyxy.cpu().tolist())]
        records.append({'image': relative.as_posix(), 'width': int(result.orig_shape[1]),
                        'height': int(result.orig_shape[0]), 'annotated_image': rendered.relative_to(output).as_posix(),
                        'detections': detections})
    write_json(output / 'detections.json', records)
    complete = {**plan, 'device': device, 'status': 'completed',
                'elapsed_seconds_including_image_output': time.perf_counter() - started,
                'timing_note': 'single command wall time, not a benchmark or FPS claim'}
    write_json(output / 'run.json', complete)
    return complete


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == 'predict':
            result = prediction(args)
        else:
            if args.command == 'val':
                weights, _ = checkpoint(args)
                output = new_output(args.output)
            with verified_data(args.data, args.data_root, args.allow_legacy_data) as (runtime, report, cfg):
                if args.command == 'check-data':
                    result = {**report, 'status': 'checked_no_model_loaded'}
                    if args.write_config:
                        import yaml
                        target = Path(args.write_config).expanduser().resolve()
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open('x', encoding='utf-8') as handle:
                            yaml.safe_dump(cfg, handle, sort_keys=False)
                        result['canonical_config'] = str(target)
                else:
                    # Original runner preserves AP .001, NMS .7, max_det 300 and all safeguards.
                    from run import main as original_run
                    options = ['val', '--model', str(weights), '--data', str(runtime),
                               '--project', str(output.parent), '--name', output.name,
                               '--device', args.device, '--imgsz', str(args.imgsz),
                               '--batch', str(args.batch), '--split', args.split]
                    if args.allow_legacy_data:
                        options.append('--allow-legacy-data')
                    if args.dry_run:
                        options.append('--dry-run')
                    else:
                        # Register before loading CBAM .pt; never patch installed package files.
                        from helmet_cbam import register_cbam
                        register_cbam()
                    evaluated = original_run(options)
                    result = {**report, 'status': 'dry_run' if args.dry_run else 'completed',
                              'output': str(output)}
                    if not args.dry_run:
                        import shutil
                        from workflow import write_json
                        shutil.copyfile(runtime, Path(evaluated) / 'runtime_data.yaml')
                        write_json(Path(evaluated) / 'data_resolution.json', report)
                        write_json(Path(evaluated) / 'portable_command.json', vars(args))
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return result
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        p.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
