"""Independent YOLOv8n-P2 + P3-CBAM training, with audited initialization and strict resume.

Default action checks only. Formal50epoch training requires explicit --train on Windows CUDA.
CPU is supported only for preflight checks. Hyperparameters are not CLI overrides.
"""
import argparse
import csv
import shutil
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from workflow import ROOT, sha256, verify_dataset, write_json, register_training

PROTOCOL = HERE/'protocol.json'
DEFAULT_NAME = 'helmet_p2_p3_cbam_isolated_640_windows'
CODE_FILES = ['tools/p3_cbam_p2/spec.json', 'tools/p3_cbam_p2/train.py', 'tools/p3_cbam_p2/model.py', 'tools/p3_cbam_p2/check.py', 'tools/p3_cbam_p2/helmet_yolov8n_p2_p3_cbam.yaml',
              'tools/p3_cbam_p2/sources.json', 'tools/p3_cbam_p2/protocol.json',
              'helmet_cbam.py', 'helmet_models.py', 'workflow.py', 'models/configs/helmet_yolov8n_p2.yaml', 'models/configs/sources.json']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def protocol():
    p = read_json(PROTOCOL)
    # Independently frozen original training-args subset, not library defaults.
    reference = read_json(HERE/'reference_parameters.json')
    if p['train'] != reference['train'] or p['historical_reference']['args_sha256'] != reference['args_sha256']:
        raise ValueError('训练参数与既有对照记录不同；不能按公平对照启动')
    if p['dataset_manifest_sha256'] != '5781ecdffa173010f539189b15565220f5e2ac597b5f8ea72f5cf7b1e8bf63e4':
        raise ValueError('数据版本不是既有对照版本')
    return p


def code_identity():
    return {p: sha256(ROOT/p) for p in CODE_FILES + ['tools/p3_cbam_p2/reference_parameters.json']}


def prepare_data(p):
    """Relocate YAML only; verify original image/label bytes and isolation evidence."""
    import yaml
    base = ROOT/p['dataset_relative']
    if not (base/'split_manifest.csv').is_file():
        raise FileNotFoundError(f'请把更新包安装到原Windows项目：缺少 {base}/split_manifest.csv')
    if sha256(base/'split_manifest.csv') != p['dataset_manifest_sha256']:
        raise ValueError('Windows数据清单与既有对照训练不一致')
    # Write a separate portable YAML; never edit original data.yaml or split manifests.
    target = ROOT/'project_work/p3_cbam_p2_training/data_local.yaml'
    cfg = {'path': str(base.resolve()), 'train': 'images/train', 'val': 'images/val',
           'test': 'images/test', 'names': {0: 'hat', 1: 'person'}}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    identity = verify_dataset(target)
    if identity.get('manifest_sha256') != p['dataset_manifest_sha256']:
        raise ValueError('核查后的数据身份不匹配')
    return target, identity


def check_parameters(actual, expected):
    different = {k: {'expected': v, 'actual': actual.get(k)} for k, v in expected.items()
                 if k not in actual or actual[k] != v}
    if different:
        raise ValueError('公平对照参数不一致: ' + json.dumps(different, ensure_ascii=False))


def check_runtime(device, p):
    import torch
    import torchvision
    import ultralytics
    actual = {'python': platform.python_version(), 'system': platform.system(),
              'torch': torch.__version__, 'torchvision': torchvision.__version__,
              'ultralytics': ultralytics.__version__, 'device': device,
              'cuda_runtime': torch.version.cuda}
    for name, version in p['runtime_required'].items():
        if actual[name].split('+')[0] != version:
            raise RuntimeError(f'{name}版本需与既有对照一致：{version}，当前 {actual[name]}；请使用原训练环境')
    if device == '0':
        reference = p['historical_reference']['environment']
        for key in ('python', 'system', 'torch', 'ultralytics'):
            if actual[key] != reference[key]:
                raise RuntimeError(f'公平对照要求使用原Windows环境：{key}={reference[key]}，当前 {actual[key]}')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA不可用，已停止；不会自动改用CPU训练')
        torch.ones((16, 16), device='cuda:0').square().sum().item()
        torchvision.ops.nms(torch.tensor([[0., 0., 10., 10.]], device='cuda:0'),
                            torch.tensor([.9], device='cuda:0'), .5)
        torch.cuda.synchronize()
        actual.update(gpu=torch.cuda.get_device_name(0),
                      vram_bytes=torch.cuda.get_device_properties(0).total_memory)
    from importlib.metadata import distributions
    actual['packages'] = {d.metadata['Name']: d.version for d in distributions() if d.metadata['Name']}
    return actual


def validate_resume(checkpoint, prior, p, identity, files, variant):
    """Reject stripped/completed/foreign checkpoints rather than starting afresh."""
    if prior['dataset'] != identity or prior['code_identity'] != files or prior['variant'] != variant:
        raise ValueError('续训的数据、代码或结构与原运行不一致')
    if prior['protocol'] != p:
        raise ValueError('续训协议发生变化')
    ckpt_args = checkpoint.get('train_args') or {}
    check_parameters(ckpt_args, p['train'])
    epoch = checkpoint.get('epoch', -1)
    if not isinstance(epoch, int) or not 0 <= epoch < p['train']['epochs'] - 1:
        raise ValueError('该权重已结束训练或缺少有效轮次，不能续训；请勿使用best.pt')
    if not (checkpoint.get('optimizer') or {}).get('state') or checkpoint.get('ema') is None:
        raise ValueError('last.pt缺少优化器/EMA，无法完整续训')
    if not checkpoint.get('scaler') or checkpoint.get('version') != p['runtime_required']['ultralytics']:
        raise ValueError('续训缺少scaler或版本不一致')
    return epoch + 1


def check_history(path, completed):
    with path.open(encoding='utf-8-sig', newline='') as f:
        epochs = [int(row['epoch']) for row in csv.DictReader(f)]
    if epochs != list(range(1, completed + 1)):
        raise ValueError('results.csv与last.pt轮次不一致，请先核对中断时的保存情况')


def trainer_class(output, data):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class AuditedTrainer(DetectionTrainer):
        def _build_train_pipeline(self):
            # Ultralytics may halve batch after OOM. Never continue a changed ablation.
            if self.batch_size != 2 or self.args.batch != 2:
                raise ValueError('显存不足或batch被改变：公平对照固定batch=2，停止而不缩小batch继续训练')
            return super()._build_train_pipeline()

        def check_resume(self, overrides):
            super().check_resume(overrides)
            # Only paths relocate. Optimizer, scheduler and epoch come from last.pt.
            self.args.data = str(data)
            self.args.project = str(output.parent)
            self.args.name = output.name
            self.args.save_dir = str(output)
    return AuditedTrainer


def check_structure(model, variant, device):
    from tools.p3_cbam_p2.check import inspect_model
    return inspect_model(model, device=device, backward=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', '--dry-run', action='store_true', dest='check',
                        help='核查数据、环境、初始化和640前向；不训练')
    parser.add_argument('--device', choices=['0', 'cpu'], default='0')
    parser.add_argument('--resume', nargs='?', const='auto', help='续训last.pt；不填写路径时使用默认实验目录')
    parser.add_argument('--name', help='新实验文件夹名；不会覆盖已有文件夹')
    parser.add_argument('--train', action='store_true', help='明确启动50轮正式训练；默认只检查')
    args = parser.parse_args(argv)
    if args.train and args.check:
        parser.error('--train不能与--check同时使用')
    args.check = not args.train
    if args.device == 'cpu' and not args.check:
        parser.error('正式公平对照使用原Windows CUDA环境；CPU只允许 --check')
    if args.name and not re.fullmatch(r'[A-Za-z0-9_-]+', args.name):
        parser.error('--name只能包含英文字母、数字、下划线或短横线')
    p = protocol()
    files = code_identity()
    environment = check_runtime(args.device, p)
    data, identity = prepare_data(p)
    variant = 'p2_p3_cbam'
    name = args.name or DEFAULT_NAME
    output = (ROOT/'runs'/name).resolve()
    from helmet_cbam import register_cbam
    from tools.p3_cbam_p2.model import build_model, validate_config
    from helmet_models import build_initialized_model, VARIANTS, tensor_digest
    register_cbam()
    validate_config()
    prior, completed, checkpoint_hash, initialization = None, 0, None, None
    if args.resume:
        path = output/'weights/last.pt' if args.resume == 'auto' else Path(args.resume).expanduser().resolve()
        if path.name != 'last.pt' or not path.is_file() or not path.is_relative_to(ROOT):
            raise ValueError('续训需指定本项目内存在的 weights/last.pt')
        output = path.parent.parent
        if args.name and output.name != args.name:
            raise ValueError('--name与续训目录不一致')
        prior = read_json(output/'experiment_identity.json')
        from ultralytics import YOLO
        model = YOLO(str(path), task='detect')
        completed = validate_resume(model.ckpt, prior, p, identity, files, variant)
        check_history(output/'results.csv', completed)
        checkpoint_hash = sha256(path)
    else:
        if output.exists():
            raise FileExistsError(f'结果目录已存在，拒绝覆盖：{output}；中断后请使用--resume')
        model, initialization = build_model(seed=p['train']['seed'])
    structure = check_structure(model, variant, args.device)
    plan = {'status': 'preflight_passed_no_training', 'variant': variant, 'protocol': p,
            'environment': environment, 'dataset': identity, 'data': str(data),
            'output': str(output), 'code_identity': files, 'structure': structure,
            'completed_epochs': completed, 'checkpoint_sha256': checkpoint_hash,
            'initialization': initialization}
    write_json(ROOT/'project_work/p3_cbam_p2_training'/f'preflight_{variant}.json', plan)
    print(json.dumps({k: plan[k] for k in ('status', 'variant', 'output', 'completed_epochs', 'structure')}, ensure_ascii=False, indent=2))
    if args.check:
        print('核查通过，没有开始训练。')
        return plan
    expected_parameters = tensor_digest(model.model.named_parameters()) if not args.resume else None
    experiment = prior or {'variant': variant, 'protocol': p, 'dataset': identity,
                           'code_identity': files, 'initialization': initialization,
                           'created_at': datetime.now(timezone.utc).isoformat()}
    # Reserve the new run exclusively; all future writes go to this explicit directory.
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'experiment_identity.json', experiment)
        write_json(output/'initialization.json', initialization)
        snapshot = output/'configuration'
        snapshot.mkdir()
        for name in ['helmet_yolov8n_p2_p3_cbam.yaml','protocol.json','reference_parameters.json','sources.json','model.py','train.py','check.py','spec.json']:
            shutil.copy2(HERE/name, snapshot/name)
    def verify_start(trainer):
        check_parameters(vars(trainer.args), p['train'])
        if Path(trainer.save_dir).resolve() != output or Path(trainer.args.data).resolve() != data.resolve():
            raise ValueError('训练器修改了输出或数据路径')
        if trainer.start_epoch != completed:
            raise ValueError('恢复的训练轮次错误')
        if expected_parameters and tensor_digest(trainer.model.named_parameters()) != expected_parameters:
            raise ValueError('训练器重建后初始参数改变，已在首次更新前停止')
        if bool(trainer.amp) != bool(p['train']['amp']):
            raise ValueError('AMP未按既有对照协议启用，拒绝悄悄改变精度模式')
        if code_identity() != files:
            raise ValueError('启动期间代码被修改')
        event = {'time': datetime.now(timezone.utc).isoformat(), 'resume': bool(args.resume),
                 'completed_epochs': completed, 'checkpoint_sha256': checkpoint_hash,
                 'environment': environment, 'effective_args': vars(trainer.args),
                 'all_locked_parameters_verified': True,
                 'initial_parameters_verified': bool(expected_parameters),
                 'runtime_amp': bool(trainer.amp), 'runtime_batch': trainer.batch_size,
                 'gradient_accumulation_initial': trainer.accumulate}
        events = read_json(output/'launch_events.json') if (output/'launch_events.json').is_file() else []
        write_json(output/'launch_events.json', events + [event])
        write_json(output/'effective_args.json', vars(trainer.args))
        write_json(output/'input_identity.json', {'dataset': identity, 'variant': variant})
        write_json(output/'environment.json', environment)
    model.add_callback('on_train_start', verify_start)
    options = {**p['train'], **p['operational'], 'device': '0', 'data': str(data),
               'project': str(output.parent), 'name': output.name, 'save_dir': str(output)}
    model.train(trainer=trainer_class(output, data),
                resume=str(path) if args.resume else False, **options)
    if verify_dataset(data) != identity or code_identity() != files:
        raise ValueError('训练期间数据或代码改变，拒绝登记为有效实验')
    if not (output/'weights/best.pt').is_file():
        raise FileNotFoundError('未产生best.pt，不登记为完成')
    # Separate registry: does not switch the demo model or overwrite old training records.
    register_training(output/'weights/best.pt', data, identity, environment,
                      vars(model.trainer.args), ROOT/'runs'/f'latest_{variant}_experiment.json')
    write_json(output/'completed.json', {'time': datetime.now(timezone.utc).isoformat(),
               'variant': variant, 'best_sha256': sha256(output/'weights/best.pt'),
               'test_evaluated': False, 'note': 'Best selected on validation; no new blind test claim'})
    print(f'训练完成：{output}。请保留整个目录用于比较。')
    return output


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n训练已中断。续训使用完整last.pt；当前未保存轮次的进度不计入。')
        sys.exit(130)
    except Exception as exc:
        print(f'\n停止：{exc}', file=sys.stderr)
        sys.exit(1)
