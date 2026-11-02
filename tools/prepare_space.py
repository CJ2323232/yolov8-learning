"""Create an allowlisted, local-only Space package; never upload or overwrite."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflow import sha256
FILES = ('app.py', 'helmet_detection.py', 'helmet_inference.py', 'helmet_fusion.py',
         'helmet_cbam.py', 'workflow.py', 'requirements.txt', 'requirements-gradio.txt',
         'LICENSE', 'THIRD_PARTY_NOTICES.md', 'models/fusion_config.json',
         'models/release_manifest.json', 'models/MODEL_CARD.md', 'docs/license_audit.md',
         'tools/third_party/weighted_boxes_fusion.py', 'tools/third_party/LICENSE-WBF',
         'tools/third_party/WBF_SOURCE.json', 'deployment/hf/start.py')
RENAMES = {'deployment/hf/Dockerfile': 'Dockerfile',
           'deployment/hf/README.md': 'README.md',
           'deployment/hf/.dockerignore': '.dockerignore'}


def prepare(output, *, with_weights=False, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output already exists; choose a new directory.')
    if not output.is_relative_to(root/'project_work'):
        raise ValueError('Use a new directory inside project_work for the deployment package.')
    pairs = [(name, name) for name in FILES] + list(RENAMES.items())
    if with_weights:
        specs = json.loads((root/'models/release_manifest.json').read_text())['models']
        for spec in specs:
            source = Path(spec['source'])
            if source.is_absolute() or not (root/source).resolve().is_relative_to(root/'run'):
                raise ValueError('Unreviewed checkpoint location.')
            if sha256(root/source) != spec['sha256']:
                raise ValueError(f'Checkpoint hash mismatch: {spec["id"]}')
            pairs.append((source.as_posix(), source.as_posix()))
    # Validate every input before creating output; never recursively copy the repository.
    for source, _ in pairs:
        if not (root/source).is_file():
            raise FileNotFoundError(source)
    output.mkdir(parents=True)
    inventory = []
    for source, target in pairs:
        destination = output/target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/source, destination)
        digest = sha256(destination)
        if digest != sha256(root/source):
            raise ValueError(f'Copy verification failed: {target}')
        inventory.append({'source': source, 'file': target,
                          'bytes': destination.stat().st_size, 'sha256': digest})
    report = {'status': 'local_staging_not_deployed', 'weights_included': with_weights,
              'publication_conditions': 'Checkpoint redistribution and metadata privacy remain under review; keep staging private.',
              'dataset_or_example_photos_included': False, 'files': inventory}
    (output/'DEPLOYMENT_MANIFEST.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--with-weights', action='store_true', help='local private staging only; does not grant publication rights')
    args = parser.parse_args(argv)
    report = prepare(args.output, with_weights=args.with_weights)
    print(json.dumps({'status': report['status'], 'files': len(report['files']),
                      'bytes': sum(x['bytes'] for x in report['files']),
                      'weights_included': report['weights_included']}))


if __name__ == '__main__':
    # Also support direct execution from the project root.
    main()
