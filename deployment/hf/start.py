"""Cloud launch adapter; reuses the existing Gradio app and trained weights."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def port_number(value):
    value = int(value)
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError('port must be between 1 and 65535')
    return value


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default=os.environ.get('HOST', '127.0.0.1'))
    parser.add_argument('--port', type=port_number, default=os.environ.get('PORT', '7860'))
    parser.add_argument('--check', action='store_true', help='verify models without loading or serving them')
    parser.add_argument('--share', action='store_true', help='explicitly create a temporary public Gradio link; keep this computer running')
    return parser.parse_args(argv)


def launch_settings(args):
    return dict(server_name=args.host, server_port=args.port, inbrowser=False, share=args.share,
                max_file_size='12mb', ssr_mode=False,
                blocked_paths=[str(ROOT/name) for name in ('models', 'run', 'runs', 'runs2',
                    'datasets', 'helmet_dataset', 'helmet_dataset_clean', 'reviewed_legacy_test',
                    'VOC2028', 'backups', 'error_analysis', 'deployment')])


def main(argv=None):
    args = arguments(argv)
    os.environ['YOLO_ONLINE_DEMO'] = '1'
    os.environ.setdefault('YOLO_DEVICE', 'cpu')
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
    os.environ.setdefault('GRADIO_TEMP_DIR', str(ROOT/'project_work/gradio_cache'))
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'project_work/matplotlib'))
    if args.share:
        os.environ.setdefault('HF_HOME', str(ROOT/'project_work/gradio_share_cache/huggingface'))
    sys.path.insert(0, str(ROOT))
    from app import demo, service, CHOICES, DEFAULT
    if not CHOICES:
        raise SystemExit('No verified model is available. Supply the reviewed checkpoints before deployment.')
    if args.check:
        print(json.dumps({'status': 'models_verified_not_loaded', 'choices': CHOICES,
                          'default': DEFAULT, 'device': service.device,
                          'loaded_models': len(service.models)}, ensure_ascii=False))
        return
    # Gradio reads this property when creating the server lifespan.
    demo.delete_cache = (600, 3600)
    if args.share:
        print('临时公开演示：无需账号；这台电脑、网络和本终端必须保持运行。关闭终端后分享失效。', flush=True)
    demo.launch(**launch_settings(args))


if __name__ == '__main__':
    main()
