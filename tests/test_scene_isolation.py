import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from split_dataset import isolation_plan, prepare_review_isolation
from workflow import sha256, verify_dataset


class SceneIsolationTests(unittest.TestCase):
    def test_risk_buckets_do_not_rewrite_human_different_or_expand_confirmed_scenes(self):
        images = [{'index': i, 'filename': f'{i}.jpg', 'split': s}
                  for i, s in enumerate(['train', 'val', 'test', 'val', 'test'])]
        pairs = [{'pair_id': 'A', 'a': 0, 'b': 1}, {'pair_id': 'B', 'a': 1, 'b': 2}]
        decisions = {'A': {'decision': 'same'}, 'B': {'decision': 'different'}}
        splits, buckets, scenes, constraints = isolation_plan(images, pairs, decisions)
        self.assertEqual(splits[0], splits[2])
        self.assertEqual(buckets[0], buckets[2])
        self.assertEqual(scenes[0], scenes[1])
        self.assertNotEqual(scenes[0], scenes[2])
        self.assertEqual(decisions['B']['decision'], 'different')
        self.assertEqual(len(constraints), 2)

    def test_contradictory_human_decisions_fail_instead_of_silent_override(self):
        images = [{'index': i, 'filename': f'{i}.jpg', 'split': s}
                  for i, s in enumerate(['train', 'val', 'test', 'val', 'test'])]
        pairs = [{'pair_id': 'A', 'a': 0, 'b': 1}, {'pair_id': 'B', 'a': 1, 'b': 2},
                 {'pair_id': 'C', 'a': 0, 'b': 2}]
        with self.assertRaisesRegex(ValueError, '矛盾'):
            isolation_plan(images, pairs, {'A': {'decision': 'same'},
                'B': {'decision': 'same'}, 'C': {'decision': 'different'}})

    def test_part_a_is_a_source_hold_in_not_a_confirmed_scene(self):
        images = [{'index': i, 'filename': f'PartA_{i}.jpg' if i < 2 else f'{i}.jpg',
                   'split': s} for i, s in enumerate(['train', 'val', 'val', 'test'])]
        splits, buckets, scenes, _ = isolation_plan(images, [], {})
        self.assertEqual(splits[0], 'train')
        self.assertEqual(splits[1], 'train')
        self.assertEqual(buckets[0], buckets[1])
        self.assertNotEqual(scenes[0], scenes[1])

    def test_versioned_copy_is_independent_and_tampered_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, review, output = root/'source', root/'review', root/'out'
            source.mkdir(); review.mkdir()
            rows, images = [], []
            for i, split in enumerate(['train']*6 + ['val']*3 + ['test']*3):
                image = f'images/{split}/{i}.png'; label = f'labels/{split}/{i}.txt'
                (source/image).parent.mkdir(parents=True, exist_ok=True)
                (source/label).parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (20, 20), (i*10, 1, 2)).save(source/image)
                (source/label).write_text('0 0.5 0.5 0.2 0.2\n')
                row = {'split': split, 'image': image, 'label': label,
                       'image_sha256': sha256(source/image), 'label_sha256': sha256(source/label),
                       'source_image_sha256': sha256(source/image), 'group_id': str(i)}
                rows.append(row)
                images.append({**row, 'filename': f'{i}.png', 'index': i})
            with (source/'split_manifest.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            digest = sha256(source/'split_manifest.csv')
            (source/'dataset_audit.json').write_text(json.dumps({'manifest_sha256': digest,
                'cross_split_exact_duplicates': 0, 'difficult_policy': 'include_as_ordinary',
                'grouping': 'exact_content_only'}))
            (source/'data.yaml').write_text(yaml.safe_dump({'path': str(source),
                'train': 'images/train', 'val': 'images/val', 'test': 'images/test',
                'names': {0: 'hat', 1: 'person'}}))
            (review/'images.json').write_text(json.dumps(images))
            (review/'pairs.json').write_text(json.dumps([{'pair_id': 'A', 'a': 0, 'b': 6}]))
            (review/'review_decisions.json').write_text(json.dumps({'dataset_manifest_sha256': digest,
                'pairs_sha256': sha256(review/'pairs.json'), 'revision': 1,
                'decisions': {'A': {'decision': 'same', 'note': ''}}}))
            (review/'summary.json').write_text(json.dumps({'images_sha256': sha256(review/'images.json')}))
            (review/'protocol.json').write_text('{}')
            report = prepare_review_isolation(source, review, output)
            identity = verify_dataset(output/'data.yaml')
            self.assertEqual(identity['scene_independence'], 'retrieved_risks_isolated_not_exhaustive')
            self.assertEqual(report['moved_images'], 1)
            self.assertFalse((output/'BUILD_INCOMPLETE.json').exists())
            with self.assertRaisesRegex(ValueError, '独立的新目录'):
                prepare_review_isolation(source, review, output)
            before = (source/'labels/val/6.txt').read_bytes()
            (output/'labels/train/6.txt').write_text('1 0.5 0.5 0.2 0.2\n')
            self.assertEqual((source/'labels/val/6.txt').read_bytes(), before)
            with self.assertRaisesRegex(ValueError, '变化'):
                verify_dataset(output/'data.yaml')
            (output/'labels/train/6.txt').write_bytes(before)
            (output/'isolation_constraints.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, '隔离证据'):
                verify_dataset(output/'data.yaml')

    def test_dry_run_never_imports_model_or_changes_training_record(self):
        import run
        identity = {'status': 'other_dataset'}
        with patch.object(run, 'verify_dataset', return_value=identity), \
                patch.dict('sys.modules', {'ultralytics': None, 'torch': None}):
            plan = run.main(['train', '--data', 'fixture.yaml', '--dry-run'])
        self.assertEqual(plan['dataset'], identity)
        self.assertEqual(plan['status'], 'preflight_only_no_training_or_prediction')

    def test_training_profiles_change_only_resolution_and_output_name(self):
        import train_helmet
        with patch.object(train_helmet, 'run', side_effect=lambda x: x):
            first = train_helmet.main(['--profile', 'baseline640', '--dry-run'])
            second = train_helmet.main(['--profile', 'highres960', '--dry-run'])
        def options(args):
            return dict(zip(args[1:-1:2], args[2:-1:2]))
        a, b = options(first), options(second)
        self.assertEqual(a.pop('--imgsz'), '640')
        self.assertEqual(b.pop('--imgsz'), '960')
        a.pop('--name'); b.pop('--name')
        self.assertEqual(a, b)
        self.assertEqual(a['--batch'], '2')


if __name__ == '__main__':
    unittest.main()
