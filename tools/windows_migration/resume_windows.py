"""Resume an exported, audited training run on Windows; never start fresh implicitly."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside transfer folder: {relative}')
    return path


def validate_checkpoint(checkpoint, expected):
    epoch = checkpoint.get('epoch', -1)
    actual = checkpoint.get('train_args') or {}
    if not isinstance(epoch, int) or not 0 <= epoch < expected['epochs'] - 1:
        raise ValueError('Checkpoint is finished or has no resumable epoch. Do not restart automatically.')
    if not (checkpoint.get('optimizer') or {}).get('state') or checkpoint.get('ema') is None:
        raise ValueError('Missing optimizer or EMA; this is not a full resume checkpoint.')
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f'Training protocol mismatch: {key}: {actual.get(key)!r} != {value!r}')
    return epoch + 1


def validate_csv(path, completed):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    epochs = [int(row['epoch']) for row in rows]
    if epochs != list(range(1, completed + 1)):
        raise ValueError('results.csv and checkpoint epochs differ; inspect before continuing.')


def portable_config(root, manifest):
    import yaml
    original = contained(root, manifest['data'])
    cfg = yaml.safe_load(original.read_text(encoding='utf-8'))
    cfg['path'] = str(original.parent.resolve())
    return cfg


def verify_export(root, manifest):
    from workflow import sha256, verify_dataset
    for relative, digest in manifest['files'].items():
        path = contained(root, relative)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Transfer file missing or changed: {relative}')
    # The source YAML remains unchanged as provenance. Only its local root is rebased.
    import yaml
    with tempfile.TemporaryDirectory(prefix='helmet_data_check_') as temp:
        config = Path(temp) / 'data.yaml'
        config.write_text(yaml.safe_dump(portable_config(root, manifest), allow_unicode=True), encoding='utf-8')
        identity = verify_dataset(config)
    prior = read_json(contained(root, manifest['source_run']) / 'input_identity.json')
    if identity != manifest['dataset'] or prior['dataset'] != identity:
        raise ValueError('Dataset identity differs from the Mac training run.')
    return identity, prior


def cuda_environment():
    import torch
    import torchvision
    import ultralytics
    if ultralytics.__version__ != '8.4.164' or torch.__version__.split('+')[0] != '2.14.0':
        raise ValueError('Keep ultralytics=8.4.164 and torch=2.14.0 for this resume.')
    if torchvision.__version__.split('+')[0] != '0.29.0':
        raise ValueError('Keep torchvision=0.29.0 paired with torch=2.14.0.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Check the NVIDIA driver and CUDA PyTorch installation; CPU fallback is disabled.')
    # A real kernel and NMS check catches drivers/builds that import but cannot execute.
    torch.ones((16, 16), device='cuda').square().sum().item()
    torchvision.ops.nms(torch.tensor([[0., 0., 10., 10.]], device='cuda'),
                        torch.tensor([0.9], device='cuda'), 0.5)
    torch.cuda.synchronize()
    try:
        driver = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        driver = 'nvidia-smi unavailable'
    return {'python': platform.python_version(), 'system': platform.system(),
            'machine': platform.machine(), 'torch': torch.__version__,
            'torchvision': torchvision.__version__, 'ultralytics': ultralytics.__version__,
            'selected_device': '0', 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0),
            'vram_bytes': torch.cuda.get_device_properties(0).total_memory,
            'cuda_available': True, 'nvidia_smi': driver,
            'migration_note': 'Epoch-boundary resume from Mac MPS to Windows CUDA; not bitwise equivalent.'}


def portable_trainer_class(output):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class PortableTrainer(DetectionTrainer):
        def check_resume(self, overrides):
            super().check_resume(overrides)
            # Only relocate output paths; the remaining protocol comes from the checkpoint.
            self.args.project = str(output.parent)
            self.args.name = output.name
            self.args.save_dir = str(output)

    return PortableTrainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify transfer, dataset, checkpoint and CUDA; do not train.')
    args = parser.parse_args(argv)
    from workflow import sha256, verify_dataset, write_json, register_training
    import torch
    from ultralytics import YOLO

    manifest = read_json(ROOT / 'transfer_manifest.json')
    identity, prior = verify_export(ROOT, manifest)
    source = contained(ROOT, manifest['source_run'])
    output = contained(ROOT, manifest['output_run'])
    checkpoint_path = output / 'weights/last.pt' if (output / 'weights/last.pt').is_file() else source / 'weights/last.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    completed = validate_checkpoint(checkpoint, manifest['training_protocol'])
    if checkpoint.get('version') != '8.4.164':
        raise ValueError('Unexpected checkpoint Ultralytics version.')
    if checkpoint_path.parent.parent == output:
        if read_json(output / 'input_identity.json') != prior:
            raise ValueError('Windows run no longer matches the exported initialization/data identity.')
    else:
        if completed != manifest['completed_epochs']:
            raise ValueError('Source checkpoint epoch differs from transfer manifest.')
    validate_csv(checkpoint_path.parent.parent / 'results.csv', completed)
    environment = cuda_environment()
    print(json.dumps({'completed_epochs': completed, 'next_epoch': completed + 1,
                      'total_epochs': manifest['training_protocol']['epochs'],
                      'checkpoint': str(checkpoint_path), 'output': str(output),
                      'environment': environment}, ensure_ascii=False, indent=2))
    if args.check:
        print('CHECK PASSED. No training started. Run 02_resume_training.bat to continue.')
        return

    # Use a separate destination so the Mac checkpoint and its records stay immutable.
    if not output.exists():
        shutil.copytree(source, output)
    elif checkpoint_path.parent.parent != output:
        raise ValueError('Destination already exists without a resumable checkpoint. Inspect it first.')
    data = contained(ROOT, manifest['data']).parent / 'data_windows.yaml'
    import yaml
    data.write_text(yaml.safe_dump(portable_config(ROOT, manifest), allow_unicode=True), encoding='utf-8')
    if verify_dataset(data) != identity:
        raise ValueError('Rebased Windows dataset differs from source.')

    checkpoint_digest = sha256(checkpoint_path)
    model = YOLO(str(checkpoint_path))

    def record_start(trainer):
        if trainer.start_epoch != completed or Path(trainer.save_dir).resolve() != output:
            raise ValueError('Trainer did not restore the expected epoch/output path.')
        if Path(trainer.args.data).resolve() != data.resolve():
            raise ValueError('Trainer did not use the rebased Windows dataset.')
        for key, value in manifest['training_protocol'].items():
            if getattr(trainer.args, key) != value:
                raise ValueError(f'Trainer changed {key}; stopping before updates.')
        write_json(output / 'input_identity.json', prior)
        write_json(output / 'environment.json', environment)
        write_json(output / 'effective_args.json', vars(trainer.args))
        write_json(output / 'command.json', {'entry': 'resume_windows.py', 'resume': True,
                    'model': str(checkpoint_path), 'data': str(data), 'device': '0'})
        events_path = output / 'migration_events.json'
        events = read_json(events_path) if events_path.is_file() else []
        events.append({'time': datetime.now(timezone.utc).isoformat(),
                       'checkpoint_sha256': checkpoint_digest, 'completed_epochs': completed,
                       'next_epoch': completed + 1, 'source_environment': manifest['source_environment'],
                       'resumed_environment': environment, 'runtime_amp_enabled': bool(trainer.amp),
                       'dataset': identity, 'strict_bitwise_reproducibility': False})
        write_json(events_path, events)

    model.add_callback('on_train_start', record_start)
    model.train(trainer=portable_trainer_class(output), resume=True, data=str(data), device='0',
                imgsz=manifest['training_protocol']['imgsz'],
                batch=manifest['training_protocol']['batch'], workers=0,
                save_dir=str(output))
    if verify_dataset(data) != identity:
        raise ValueError('Dataset changed during training; refuse to register.')
    register_training(output / 'weights/best.pt', data, identity, environment, vars(model.trainer.args))
    print(f'Training finished. Results: {output}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted. Saved completed epochs remain available; use 02_resume_training.bat again.')
        sys.exit(130)
