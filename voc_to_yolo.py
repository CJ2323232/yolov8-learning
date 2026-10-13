"""Convert VOC labels, preserving difficult/truncated metadata and the chosen protocol."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

from workflow import ROOT, write_json

CLASS_MAP = {'hat': 0, 'person': 1}


def convert_annotation(xml_path):
    root = ET.parse(xml_path).getroot()
    width, height = int(root.findtext('size/width')), int(root.findtext('size/height'))
    if min(width, height) <= 0:
        raise ValueError(f'无效图片尺寸：{xml_path}')
    lines, metadata = [], []
    for index, obj in enumerate(root.findall('object')):
        name = obj.findtext('name').strip()
        difficult, truncated = int(obj.findtext('difficult', '0')), int(obj.findtext('truncated', '0'))
        if difficult not in (0, 1) or truncated not in (0, 1):
            raise ValueError(f'无效 VOC 标记：{xml_path}')
        if name not in CLASS_MAP and name != 'dog':
            raise ValueError(f'未审核的未知类别：{name}，{xml_path}')
        metadata.append({'xml': Path(xml_path).name, 'object_index': index, 'class': name,
                         'difficult': difficult, 'truncated': truncated, 'included': name in CLASS_MAP})
        if name not in CLASS_MAP:
            continue
        box = obj.find('bndbox')
        x1, y1, x2, y2 = [float(box.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f'框越界：{xml_path}')
        lines.append(f'{CLASS_MAP[name]} {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} '
                     f'{(x2-x1)/width:.6f} {(y2-y1)/height:.6f}')
    return lines, metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'VOC2028/Annotations')
    p.add_argument('--output', type=Path, default=ROOT/'VOC2028/labels')
    args = p.parse_args()
    xmls = sorted(args.source.glob('*.xml'))
    if not xmls:
        raise FileNotFoundError(args.source)
    # Validate the entire input first; unknown classes cannot leave a half-converted set.
    converted = [(xml, *convert_annotation(xml)) for xml in xmls]
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = []
    for xml, lines, meta in converted:
        (args.output/(xml.stem+'.txt')).write_text('\n'.join(lines), encoding='utf-8')
        metadata.extend(meta)
    with (args.output/'annotation_metadata.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['xml', 'object_index', 'class', 'difficult', 'truncated', 'included'])
        writer.writeheader(); writer.writerows(metadata)
    difficult = Counter(m['class'] for m in metadata if m['difficult'] and m['included'])
    write_json(args.output/'conversion_protocol.json',
               {'difficult_policy': 'include_as_ordinary', 'difficult_objects': dict(difficult),
                'coordinate_policy': 'existing SHWD continuous xyxy formula; no endpoint shift',
                'ignored_non_target_objects': sum(not m['included'] for m in metadata),
                'note': 'Not the VOC difficult-ignore evaluation protocol.'})
    print(f'已转换 {len(xmls)} 个文件；difficult 作为普通目标，原始标记已保存：{dict(difficult)}')


if __name__ == '__main__':
    main()
