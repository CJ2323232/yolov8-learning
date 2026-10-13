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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'VOC2028')
    parser.add_argument('--output', type=Path, default=ROOT/'helmet_dataset_clean')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--groups-file', type=Path, help='完整场景分组 CSV：image,group_id')
    args = parser.parse_args()
    report = prepare(args.source, args.output, args.seed, args.groups_file)
    print(json.dumps({k: v for k, v in report.items() if k not in ('conflicting_groups', 'identical_annotation_duplicate_groups')}, ensure_ascii=False, indent=2))
    print(f'新数据：{args.output}；冲突版本已隔离，原数据未改动。')


if __name__ == '__main__':
    main()
