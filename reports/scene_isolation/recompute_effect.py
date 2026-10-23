"""Recompute split-isolation effects from public metadata; no images or training required."""
import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path


def calculate():
    folder = Path(__file__).resolve().parent
    audit = json.loads((folder / 'dataset_audit.json').read_text())
    before_bytes = (folder.parent / 'clean_dataset_manifest.csv').read_bytes()
    after_bytes = gzip.decompress((folder / 'split_manifest.csv.gz').read_bytes())
    edge_bytes = gzip.decompress((folder / 'isolation_constraints.json.gz').read_bytes())
    review_bytes = (folder / 'review_decisions.json').read_bytes()
    for data, key in [(before_bytes, 'source_dataset_manifest_sha256'),
                      (after_bytes, 'manifest_sha256'),
                      (edge_bytes, 'isolation_constraints_sha256'),
                      (review_bytes, 'review_decisions_sha256')]:
        if hashlib.sha256(data).hexdigest() != audit[key]:
            raise ValueError('Evidence checksum mismatch: ' + key)
    def rows(data):
        return {Path(r['image']).name: r for r in csv.DictReader(io.StringIO(data.decode()))}
    before, after = rows(before_bytes), rows(after_bytes)
    if before.keys() != after.keys():
        raise ValueError('Image inventory changed')
    unchanged = all(before[n][k] == after[n][k] for n in before
                    for k in ('image_sha256', 'label_sha256'))
    edges = [e for e in json.loads(edge_bytes) if e['reason'] == 'retrieved_risk']
    decisions = json.loads(review_bytes)['decisions']
    def describe(mapping):
        cross = [e for e in edges if mapping[e['a']]['split'] != mapping[e['b']]['split']]
        return {'images': len(mapping),
                'splits': dict(sorted(Counter(r['split'] for r in mapping.values()).items())),
                'retrieved_candidate_cross_split_pairs': len(cross),
                'human_confirmed_same_cross_split_pairs': sum(
                    decisions.get(e['pair_id'], {}).get('decision') == 'same' for e in cross),
                'part_a_images': {s: sum(n.startswith('PartA_') and r['split'] == s
                                         for n, r in mapping.items()) for s in ('train', 'val', 'test')}}
    return {'comparison': 'first content-clean split versus conservative scene-risk isolation',
            'before': describe(before), 'after': describe(after),
            'retrieved_candidate_pairs': len(edges),
            'unique_images_in_candidates': len({e[k] for e in edges for k in ('a', 'b')}),
            'moved_images': sum(before[n]['split'] != after[n]['split'] for n in before),
            'image_and_label_hashes_unchanged_in_manifests': unchanged,
            'scope': 'Manifest comparison only; does not read image bytes, establish exhaustive scene independence, or measure model accuracy.'}


if __name__ == '__main__':
    print(json.dumps(calculate(), ensure_ascii=False, indent=2))
