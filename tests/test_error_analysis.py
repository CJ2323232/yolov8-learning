import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np

from analyze_detection_errors import match, overlap, size_bin, verify
from workflow import sha256


class ErrorAnalysisTests(unittest.TestCase):
    def test_duplicate_detection_is_not_two_true_positives(self):
        pairs,fn,fp=match([[0,0,10,10,0]],[[0,0,10,10,.9,0],[0,0,10,10,.8,0]])
        self.assertEqual([(p,g) for p,g,_ in pairs],[(0,0)])
        self.assertEqual(fn,[]);self.assertEqual(fp,[1])

    def test_perfect_overlap_wrong_class_is_false_positive_and_miss(self):
        pairs,fn,fp=match([[0,0,10,10,0]],[[0,0,10,10,.9,1]])
        self.assertEqual((pairs,fn,fp),([],[0],[0]))

    def test_uses_best_available_gt_in_crowded_scene(self):
        pairs,fn,fp=match([[0,0,10,10,0],[2,0,12,10,0]],
                          [[0,0,10,10,.9,0],[.5,0,10.5,10,.8,0]])
        self.assertEqual([(p,g) for p,g,_ in pairs],[(0,0),(1,1)])
        self.assertEqual((fn,fp),([],[]))

    def test_match_all_sizes_before_counting_small_misses(self):
        pairs,fn,fp=match([[0,0,20,20,0],[5,5,15,15,0]],[[0,0,20,20,.9,0]],threshold=.2)
        self.assertEqual([(p,g) for p,g,_ in pairs],[(0,0)])
        self.assertEqual(fn,[1])

    def test_localization_below_threshold_is_not_a_hit(self):
        pairs,fn,fp=match([[0,0,10,10,0]],[[5,0,15,10,.9,0]])
        self.assertEqual((pairs,fn,fp),([],[0],[0]))
        self.assertAlmostEqual(overlap([[0,0,10,10]],[[5,0,15,10]])[0,0],1/3)

    def test_empty_predictions_and_empty_gt(self):
        self.assertEqual(match([[0,0,10,10,0]],[]),([],[0],[]))
        self.assertEqual(match([],[[0,0,10,10,.9,0]]),([],[],[0]))
        self.assertEqual(match([],[]),([],[],[]))

    def test_size_bins_use_fixed_reference_scale_and_exact_boundaries(self):
        self.assertEqual(size_bin([0,0,31,31],640,640)[0],'small')
        self.assertEqual(size_bin([0,0,32,32],640,640)[0],'medium')
        self.assertEqual(size_bin([0,0,62,62],1280,1280)[0],'small')
        self.assertEqual(size_bin([0,0,15,15],640,640)[0],'tiny')

    def test_changed_label_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'images/test').mkdir(parents=True);(root/'labels/test').mkdir(parents=True)
            im=root/'images/test/a.jpg';im.write_bytes(b'image')
            label=root/'labels/test/a.txt';label.write_text('0 0.5 0.5 0.2 0.2')
            rows=[{'image':'a.jpg','image_sha256':sha256(im),'label_sha256':sha256(label)}]
            verify(root,rows)
            label.write_text('0 0.5 0.5 0.3 0.3')
            with self.assertRaisesRegex(ValueError,'changed'):verify(root,rows)


if __name__=='__main__':unittest.main()
