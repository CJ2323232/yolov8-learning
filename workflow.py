"""Dataset checks and explicit training-artifact selection (no model imports)."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent
TRAINING_RECORD = ROOT / 'runs/latest_helmet_training.json'
RELEASE_SHA256 = '55be821e9eefdd0a69caf4abd3fdb2a22277059e21ec155222102b32cef2e650'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Atomically replace metadata, never a checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, default=str)
            f.write('\n')
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def read_labels(path):
    """Canonical order-independent valid boxes, with six-decimal edge tolerance."""
    boxes = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = tuple(float(x) for x in line.split())
            if len(values) != 5 or not all(math.isfinite(x) for x in values):
                raise ValueError('expected five finite numbers')
            c, x, y, w, h = values
            if c not in (0, 1) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError('invalid class or normalized box')
            if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
                raise ValueError('box extends outside image')
            boxes.append(values)
        except ValueError as exc:
            raise ValueError(f'{path}:{number}: {exc}') from exc
    if len(boxes) != len(set(boxes)):
        raise ValueError(f'duplicate boxes: {path}')
    return tuple(sorted(boxes))


def verify_dataset(data, *, allow_legacy=False):
    """Verify all image/label bytes and split isolation before helmet experiments."""
    import yaml
    config = Path(data).expanduser().resolve()
    if not config.is_file():
        if str(data) == 'coco8.yaml':
            return {'status': 'library_sample_not_helmet_baseline'}
        raise FileNotFoundError(config)
    cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    names = cfg.get('names', {})
    names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    if set(names.values()) != {'hat', 'person'}:
        return {'status': 'other_dataset', 'config_sha256': sha256(config)}
    if names != {0: 'hat', 1: 'person'}:
        raise ValueError('类别映射必须为 0=hat、1=person')
    root = Path(cfg.get('path', config.parent)).expanduser()
    if not root.is_absolute():
        root = config.parent / root
    root = root.resolve()
    if (root/'BUILD_INCOMPLETE.json').exists():
        raise ValueError('数据版本尚未构建完成，不能用于训练或评价')
    manifest, report = root / 'split_manifest.csv', root / 'dataset_audit.json'
    if not manifest.is_file() or not report.is_file():
        if allow_legacy:
            return {'status': 'legacy_diagnostic_not_independent_baseline', 'config_sha256': sha256(config)}
        raise ValueError('缺少审计清单。先运行 split_dataset.py；历史诊断需显式 --allow-legacy-data。')
    audit = json.loads(report.read_text(encoding='utf-8'))
    fingerprint = sha256(manifest)
    if audit.get('manifest_sha256') != fingerprint or audit.get('cross_split_exact_duplicates') != 0:
        raise ValueError('数据审计记录无效或存在跨集合重复图片')
    if audit.get('difficult_policy') != 'include_as_ordinary':
        raise ValueError('未知 difficult 评价协议')
    with manifest.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    expected, seen_hashes, seen_groups, by_name = set(), {}, {}, {}
    for row in rows:
        split = row['split']
        if split not in ('train', 'val', 'test'):
            raise ValueError('invalid manifest split')
        image, label = root / row['image'], root / row['label']
        if not image.resolve().is_relative_to(root) or not label.resolve().is_relative_to(root):
            raise ValueError('manifest path escapes dataset')
        if image.parent != root/'images'/split or label != root/'labels'/split/(image.stem+'.txt'):
            raise ValueError('manifest split/path mismatch')
        if image in expected:
            raise ValueError('duplicate manifest entry')
        expected.add(image)
        if image.name in by_name:
            raise ValueError('数据版本内图片文件名重复')
        by_name[image.name] = row
        if sha256(image) != row['image_sha256'] or sha256(label) != row['label_sha256']:
            raise ValueError(f'数据内容已变化，不能沿用旧清单：{image.name}')
        read_labels(label)
        for mapping, key in [(seen_hashes, row['image_sha256']), (seen_groups, row['group_id'])]:
            if mapping.setdefault(key, split) != split:
                raise ValueError('cross-split content or scene leakage')
    actual = set()
    for split in ('train', 'val', 'test'):
        if cfg.get(split) != f'images/{split}':
            raise ValueError('YAML split 与审计清单不一致')
        actual.update(p for p in (root/'images'/split).rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS)
    if actual != expected or any(not any(r['split'] == s for r in rows) for s in ('train', 'val', 'test')):
        raise ValueError('数据目录与清单不一致，或划分为空')
    isolation = audit.get('grouping') == 'conservative_candidate_isolation'
    if isolation:
        for filename, key in [('review_decisions.json', 'review_decisions_sha256'),
                              ('isolation_constraints.json', 'isolation_constraints_sha256')]:
            if sha256(root/filename) != audit.get(key):
                raise ValueError('隔离证据已变化，不能沿用原数据审计')
        for edge in json.loads((root/'isolation_constraints.json').read_text()):
            a, b = by_name[edge['a']], by_name[edge['b']]
            if a['split'] != b['split'] or a['group_id'] != b['group_id']:
                raise ValueError('仍存在跨分区的候选关联，拒绝训练')
        if any(r['split'] != 'train' for name, r in by_name.items() if name.startswith('PartA_')):
            raise ValueError('PartA 来源隔离策略被破坏')
    return {'status': 'exact_duplicates_removed_conflicts_quarantined',
            'manifest_sha256': fingerprint, 'images': len(rows),
            'difficult_policy': audit['difficult_policy'],
            'scene_independence': ('retrieved_risks_isolated_not_exhaustive' if isolation else
                'user_grouped_not_manually_verified' if audit['grouping'] == 'user_scene_groups' else 'not_verified'),
            **({'isolation_constraints_sha256': audit['isolation_constraints_sha256'],
                'review_decisions_sha256': audit['review_decisions_sha256']} if isolation else {})}


def latest_training(record_path=TRAINING_RECORD):
    path = Path(record_path)
    if not path.is_file():
        raise FileNotFoundError('尚无新训练记录。请先运行 train_helmet.py；不会自动用历史模型作新基准。')
    record = json.loads(path.read_text(encoding='utf-8'))
    if sha256(record['model']) != record['model_sha256']:
        raise ValueError('已登记模型的内容发生变化，停止使用')
    return record


def select_model(explicit=None, *, require_training=False, record_path=TRAINING_RECORD):
    if explicit:
        model = Path(explicit).expanduser()
        model = model if model.is_absolute() else ROOT / model
        if not model.is_file():
            raise FileNotFoundError(model)
        return model.resolve(), 'explicit_model'
    if Path(record_path).is_file():
        record = latest_training(record_path)
        return Path(record['model']), 'registered_training'
    if require_training:
        latest_training(record_path)
    model = ROOT / 'models/helmet_yolov8n.pt'
    if not model.is_file() or sha256(model) != RELEASE_SHA256:
        raise FileNotFoundError('先运行 python download_weights.py 获取历史演示模型，或完成新训练。')
    return model, 'historical_demo_not_clean_baseline'


def register_training(model, data, identity, environment, effective_args, record_path=TRAINING_RECORD):
    if identity.get('status') != 'exact_duplicates_removed_conflicts_quarantined':
        return None
    record = {'model': str(Path(model).resolve()), 'model_sha256': sha256(model),
              'data': str(Path(data).resolve()), 'dataset': identity,
              'environment': environment, 'effective_train_args': effective_args}
    write_json(Path(model).parent.parent/'training_artifact.json', record)
    write_json(record_path, record)
    return record
