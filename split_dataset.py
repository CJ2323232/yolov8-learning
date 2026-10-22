"""Prepare a new dataset: deduplicate content; quarantine conflicting annotations."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

from workflow import IMAGE_EXTENSIONS, ROOT, read_labels, sha256, write_json
from voc_to_yolo import convert_annotation


def prepare(source, output, seed=42, groups_file=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'不会覆盖已有数据：{output}')
    images = sorted(p for p in (source/'JPEGImages').iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images or len({p.stem for p in images}) != len(images):
        raise ValueError('图片为空或存在同名不同扩展名的图片')
    scene_groups = {}
    if groups_file:
        with Path(groups_file).open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                if not row['group_id'] or row['image'] in scene_groups:
                    raise ValueError('场景分组为空或重复')
                scene_groups[row['image']] = row['group_id']
        if set(scene_groups) != {p.name for p in images}:
            raise ValueError('场景分组必须覆盖所有源图片：image,group_id')
    content_groups = defaultdict(list)
    metadata, difficult_total, unknown, repaired = [], Counter(), Counter(), []
    from PIL import Image
    for image in images:
        label = source/'labels'/(image.stem+'.txt')
        boxes = read_labels(label)
        xml = ET.parse(source/'Annotations'/(image.stem+'.xml')).getroot()
        converted, _ = convert_annotation(source/'Annotations'/(image.stem+'.xml'))
        if boxes != tuple(sorted(tuple(map(float, line.split())) for line in converted)):
            raise ValueError(f'TXT 与 XML 转换不一致：{image.name}')
        with Image.open(image) as im:
            dimensions = im.size
            im.verify()
        source_hash = sha256(image)
        repair_eoi = False
        if image.suffix.lower() in ('.jpg', '.jpeg'):
            with image.open('rb') as f:
                f.seek(-2, 2)
                repair_eoi = f.read() != b'\xff\xd9'
        # Do not re-encode pixels. The source is untouched; copies receive an EOI marker.
        fingerprint = hashlib.sha256(image.read_bytes()+b'\xff\xd9').hexdigest() if repair_eoi else source_hash
        if repair_eoi:
            repaired.append(image.name)
        if dimensions != (int(xml.findtext('size/width')), int(xml.findtext('size/height'))):
            raise ValueError(f'XML 与图片尺寸不同：{image.name}')
        for index, obj in enumerate(xml.findall('object')):
            name = obj.findtext('name').strip()
            difficult = int(obj.findtext('difficult', '0'))
            truncated = int(obj.findtext('truncated', '0'))
            metadata.append({'image': image.name, 'object_index': index, 'class': name,
                             'difficult': difficult, 'truncated': truncated})
            if name in ('hat', 'person'):
                difficult_total[name] += difficult
            elif name == 'dog':
                unknown[name] += 1
            else:
                raise ValueError(f'未审核的类别 {name}：{image.name}')
        content_groups[fingerprint].append({'image': image, 'label': label, 'boxes': boxes,
                                             'label_sha256': sha256(label), 'source_image_sha256': source_hash,
                                             'repair_eoi': repair_eoi})
    selected, conflicts, deduplicated = [], [], []
    for digest, members in sorted(content_groups.items()):
        public = [{'image': m['image'].name, 'label_sha256': m['label_sha256']} for m in members]
        if len({m['boxes'] for m in members}) > 1:
            conflicts.append({'image_sha256': digest, 'members': public})
            continue  # Never guess which annotation is correct.
        if len(members) > 1:
            deduplicated.append({'image_sha256': digest, 'members': public, 'kept': members[0]['image'].name})
        member = members[0]
        member['image_sha256'] = digest
        ids = {scene_groups[m['image'].name] for m in members} if scene_groups else {digest}
        if len(ids) != 1:
            raise ValueError('同一图片内容被指定到不同场景组')
        member['group_id'] = next(iter(ids))
        selected.append(member)
    units = defaultdict(list)
    for member in selected:
        units[member['group_id']].append(member)
    group_ids = sorted(units)
    random.Random(seed).shuffle(group_ids)
    targets = {'train': int(len(selected)*.8), 'val': int(len(selected)*.1)}
    targets['test'] = len(selected)-targets['train']-targets['val']
    partitions = {s: [] for s in targets}
    for gid in group_ids:
        split = max(targets, key=lambda s: (targets[s]-len(partitions[s]))/max(targets[s], 1))
        partitions[split].extend(units[gid])
    if any(not rows for rows in partitions.values()):
        raise ValueError('分组数量不足，无法生成三个非空划分')
    output.mkdir(parents=True)
    rows, counts = [], {}
    for split, members in partitions.items():
        (output/'images'/split).mkdir(parents=True)
        (output/'labels'/split).mkdir(parents=True)
        objects = Counter()
        for m in sorted(members, key=lambda m: m['image'].name):
            ip, lp = Path('images')/split/m['image'].name, Path('labels')/split/m['label'].name
            shutil.copy2(m['image'], output/ip)
            if m['repair_eoi']:
                with (output/ip).open('ab') as f:
                    f.write(b'\xff\xd9')
            with Image.open(output/ip) as copied:
                copied.load()
            if sha256(output/ip) != m['image_sha256']:
                raise ValueError('复制期间图片内容发生变化')
            shutil.copy2(m['label'], output/lp)
            rows.append({'split': split, 'image': ip.as_posix(), 'label': lp.as_posix(),
                         'image_sha256': m['image_sha256'], 'label_sha256': m['label_sha256'],
                         'source_image_sha256': m['source_image_sha256'],
                         'group_id': m['group_id']})
            objects.update(int(box[0]) for box in m['boxes'])
        counts[split] = {'images': len(members), 'objects': dict(objects)}
    for conflict in conflicts:
        for m in content_groups[conflict['image_sha256']]:
            for kind, path in [('images', m['image']), ('labels', m['label'])]:
                dest = output/'quarantine'/kind/path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
    with (output/'split_manifest.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (output/'annotation_metadata.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'object_index', 'class', 'difficult', 'truncated'])
        writer.writeheader(); writer.writerows(metadata)
    audit = {'schema_version': 1, 'seed': seed, 'source_images': len(images),
             'accepted_images': len(selected), 'splits': counts,
             'conflicting_groups': conflicts, 'quarantined_images': sum(len(c['members']) for c in conflicts),
             'identical_annotation_duplicate_groups': deduplicated, 'cross_split_exact_duplicates': 0,
             'difficult_policy': 'include_as_ordinary', 'source_difficult_objects': dict(difficult_total),
             'ignored_non_target_objects': dict(unknown),
             'source_jpeg_eoi_repairs': repaired,
             'grouping': 'user_scene_groups' if scene_groups else 'exact_content_only',
             'limitations': 'Near-duplicates and scene independence require manual review. Historical weights cannot establish a baseline on this new split.',
             'manifest_sha256': sha256(output/'split_manifest.csv')}
    write_json(output/'dataset_audit.json', audit)
    import yaml
    (output/'data.yaml').write_text(yaml.safe_dump({'path': str(output), 'train': 'images/train',
        'val': 'images/val', 'test': 'images/test', 'names': {0: 'hat', 1: 'person'}}, allow_unicode=True), encoding='utf-8')
    return audit


def isolation_plan(images, pairs, decisions):
    """Co-locate risk buckets, without claiming candidates are confirmed scenes.

    A bucket may deliberately include DIFFERENT scenes. Human 'different'
    decisions prevent a confirmed-scene union, not sharing a dataset partition.
    SCUT PartA is conservatively kept in training as one source family.
    """
    by_index = {m['index']: m for m in images}
    if len(by_index) != len(images) or len({m['filename'] for m in images}) != len(images):
        raise ValueError('图片索引或文件名重复')
    risk, confirmed = {i: i for i in by_index}, {i: i for i in by_index}

    def find(parents, i):
        if i not in parents:
            raise ValueError('关联引用了不存在的图片')
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(parents, a, b):
        a, b = find(parents, a), find(parents, b)
        parents[max(a, b)] = min(a, b)

    lookup = {p['pair_id']: p for p in pairs}
    if len(lookup) != len(pairs) or not set(decisions).issubset(lookup):
        raise ValueError('核查记录与候选编号不匹配')
    constraints = []
    for p in pairs:
        union(risk, p['a'], p['b'])
        constraints.append({'a': by_index[p['a']]['filename'],
                            'b': by_index[p['b']]['filename'],
                            'reason': 'retrieved_risk', 'pair_id': p['pair_id']})
    for pid, choice in decisions.items():
        if choice['decision'] not in ('same', 'different', 'unsure'):
            raise ValueError('未知人工选择')
        if choice['decision'] == 'same':
            p = lookup[pid]
            union(confirmed, p['a'], p['b'])
    for pid, choice in decisions.items():
        p = lookup[pid]
        if choice['decision'] == 'different' and find(confirmed, p['a']) == find(confirmed, p['b']):
            raise ValueError(f'人工同组关系与不同组关系矛盾：{pid}')
    part_a = [m['index'] for m in images if m['filename'].startswith('PartA_')]
    for i in part_a[1:]:
        union(risk, part_a[0], i)
        constraints.append({'a': by_index[part_a[0]]['filename'], 'b': by_index[i]['filename'],
                            'reason': 'PartA_source_family', 'pair_id': ''})
    units = defaultdict(list)
    for i in by_index:
        units[find(risk, i)].append(i)
    destinations, group_ids, scene_ids = {}, {}, {}
    for root, members in sorted(units.items()):
        counts = Counter(by_index[i]['split'] for i in members)
        # Most original assignments survive; deterministic ties favor train.
        split = 'train' if part_a and find(risk, part_a[0]) == root else max(
            ('train', 'val', 'test'), key=lambda s: counts[s])
        for i in members:
            destinations[i] = split
            group_ids[i] = f'ISOLATION_{root:05d}'
            scene_ids[i] = f'CONFIRMED_{find(confirmed, i):05d}'
    if any(not any(s == split for s in destinations.values()) for split in ('train', 'val', 'test')):
        raise ValueError('保守分组导致评价集为空；需要人工重新设计分组，不能自动拆散')
    return destinations, group_ids, scene_ids, constraints


def copy_independent(source, destination):
    """APFS copy-on-write clone, with portable independent-copy fallback.

    Never hard-link: changing a derived dataset must not mutate the source.
    """
    import ctypes
    import os
    import sys
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    if sys.platform == 'darwin':
        library = ctypes.CDLL(None, use_errno=True)
        clone = getattr(library, 'clonefile', None)
        if clone is not None:
            clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            clone.restype = ctypes.c_int
            if clone(os.fsencode(source), os.fsencode(destination), 0) == 0:
                return 'copy_on_write'
    shutil.copy2(source, destination)
    return 'independent_copy'


def prepare_review_isolation(source, review, output):
    """Make a versioned dataset with no cross-split *retrieved* risk edges."""
    from workflow import verify_dataset
    import yaml
    source, review, output = (Path(p).resolve() for p in (source, review, output))
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('输出必须为独立的新目录，不能覆盖或嵌套原数据')
    identity = verify_dataset(source/'data.yaml')
    images = json.loads((review/'images.json').read_text())
    pairs = json.loads((review/'pairs.json').read_text())
    state = json.loads((review/'review_decisions.json').read_text())
    summary = json.loads((review/'summary.json').read_text())
    if (state['dataset_manifest_sha256'] != identity['manifest_sha256']
            or state['pairs_sha256'] != sha256(review/'pairs.json')
            or summary['images_sha256'] != sha256(review/'images.json')):
        raise ValueError('核查快照身份已变化')
    with (source/'split_manifest.csv').open(newline='') as handle:
        original = list(csv.DictReader(handle))
    by_name = {Path(row['image']).name: row for row in original}
    if len(original) != len(images) or set(by_name) != {m['filename'] for m in images}:
        raise ValueError('核查图片与源清单不一致')
    for m in images:
        row = by_name[m['filename']]
        if any(row[k] != m[k] for k in ('image', 'label', 'split', 'image_sha256', 'label_sha256')):
            raise ValueError('核查图片元信息不匹配')
    destinations, buckets, scene_ids, constraints = isolation_plan(images, pairs, state['decisions'])
    output.mkdir(parents=True)
    marker = output/'BUILD_INCOMPLETE.json'
    write_json(marker, {'status': 'building'})
    rows, changes, copy_methods, counts = [], [], Counter(), {}
    object_counts = defaultdict(Counter)
    for m in images:
        row = by_name[m['filename']]
        split = destinations[m['index']]
        new_image = Path('images')/split/m['filename']
        new_label = Path('labels')/split/Path(row['label']).name
        for source_relative, target_relative, key in [
                (row['image'], new_image, 'image_sha256'),
                (row['label'], new_label, 'label_sha256')]:
            copy_methods[copy_independent(source/source_relative, output/target_relative)] += 1
            if sha256(output/target_relative) != row[key]:
                raise ValueError('派生数据复制后内容不一致')
        rows.append({**row, 'split': split, 'image': new_image.as_posix(),
                     'label': new_label.as_posix(), 'group_id': buckets[m['index']],
                     'confirmed_scene_id': scene_ids[m['index']]})
        object_counts[split].update(int(box[0]) for box in read_labels(output/new_label))
        if split != row['split']:
            changes.append({'image': m['filename'], 'old_split': row['split'],
                            'new_split': split, 'isolation_bucket': buckets[m['index']]})
    for name, data, fields in [('split_manifest.csv', rows, list(rows[0])),
                               ('split_changes.csv', changes, ['image', 'old_split', 'new_split', 'isolation_bucket'])]:
        with (output/name).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(data)
    for fn in ('review_decisions.json', 'protocol.json'):
        shutil.copy2(review/fn, output/fn)
    write_json(output/'isolation_constraints.json', constraints)
    audit = json.loads((source/'dataset_audit.json').read_text())
    counts = {split: {'images': sum(r['split'] == split for r in rows),
                     'objects': dict(object_counts[split])} for split in ('train', 'val', 'test')}
    audit.update(grouping='conservative_candidate_isolation', splits=counts,
                 seed=None, manifest_sha256=sha256(output/'split_manifest.csv'),
                 source_dataset_manifest_sha256=identity['manifest_sha256'],
                 reviewed_pair_counts=dict(Counter(d['decision'] for d in state['decisions'].values())),
                 review_decisions_sha256=sha256(output/'review_decisions.json'),
                 isolation_constraints_sha256=sha256(output/'isolation_constraints.json'),
                 isolation_policy={'candidate_edges': len(pairs),
                    'PartA_policy': 'entire source family in train; NOT a claim that all images are one scene',
                    'bucket_assignment': 'largest original split; ties train then val then test; PartA forced train',
                    'human_different_policy': 'retain decisions, never merge confirmed scene IDs; risk buckets may co-locate different scenes',
                    'candidate_cross_split_edges': 0, 'confirmed_same_cross_split_edges': 0,
                    'scene_independence_proven': False},
                 moved_images=len(changes), copy_methods=dict(copy_methods),
                 limitations='Only retrieved risks and the PartA source family are isolated; undiscovered relations may remain. '
                    'Val/test exclude classroom PartA and do not establish generalization to unseen classrooms. '
                    'Historical weights have seen members of this dataset and are not a clean baseline. '
                    'Quarantined source variants remain in the original clean dataset; none are reintroduced.')
    write_json(output/'dataset_audit.json', audit)
    (output/'data.yaml').write_text(yaml.safe_dump({'path': str(output), 'train': 'images/train',
        'val': 'images/val', 'test': 'images/test', 'names': {0: 'hat', 1: 'person'}}, allow_unicode=True))
    marker.unlink()
    verify_dataset(output/'data.yaml')
    if verify_dataset(source/'data.yaml') != identity:
        raise ValueError('源数据身份发生变化')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'VOC2028')
    parser.add_argument('--output', type=Path, default=ROOT/'helmet_dataset_clean')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--groups-file', type=Path, help='完整场景分组 CSV：image,group_id')
    parser.add_argument('--isolate-review', type=Path, help='使用核查目录构建保守隔离版本；--source 改为现有清洗数据目录')
    args = parser.parse_args()
    report = (prepare_review_isolation(args.source, args.isolate_review, args.output) if args.isolate_review
              else prepare(args.source, args.output, args.seed, args.groups_file))
    print(json.dumps({k: v for k, v in report.items() if k not in ('conflicting_groups', 'identical_annotation_duplicate_groups')}, ensure_ascii=False, indent=2))
    print(f'新数据：{args.output}；冲突版本已隔离，原数据未改动。')


if __name__ == '__main__':
    main()
