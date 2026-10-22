"""Train a fresh helmet baseline on the audited dataset; importing is safe."""
import sys
import argparse
import json
from run import main as run
from workflow import ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description='安全帽训练：--variant standard/p2，--profile baseline640/highres960；其余参数见 run.py --help')
    parser.add_argument('--profile', choices=['baseline640', 'highres960'], default='baseline640')
    parser.add_argument('--variant', choices=['standard', 'p2'], default='standard')
    profile, extra = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if any(arg.split('=')[0] in ('--model', '--pretrained') for arg in extra):
        parser.error('此入口固定 standard/P2 结构与官方初始化；自定义模型请使用 run.py')
    registry = ROOT/'datasets/current_training_dataset.json'
    data = ROOT/'helmet_dataset_clean/data.yaml'
    if registry.is_file():
        selection = json.loads(registry.read_text())
        data = ROOT/selection['data']
        from workflow import sha256
        if sha256(data.parent/'split_manifest.csv') != selection['manifest_sha256']:
            raise ValueError('当前训练数据版本与登记清单不一致')
    size = 960 if profile.profile == 'highres960' else 640
    from helmet_models import VARIANTS
    pretrained = str(ROOT/'yolov8n.pt')
    # Both future profiles share the same microbatch and optimizer settings.
    # Resolution is the only profile-level change; neither starts on import.
    return run(['train', '--model', str(VARIANTS[profile.variant]),
                '--pretrained', pretrained, '--data', str(data),
                '--epochs', '50', '--imgsz', str(size), '--batch', '2',
                '--optimizer', 'AdamW', '--lr0', '0.001667', '--seed', '0',
                '--mosaic', '1.0', '--close-mosaic', '10', '--scale', '0.5',
                '--translate', '0.1', '--mixup', '0.0', '--nbs', '64',
                '--project', str(ROOT/'runs'), '--name', f'helmet_{profile.variant}_isolated_{size}',
                *extra])


if __name__ == '__main__':
    main()
