import unittest
import numpy as np
from experiment import choose, match, finish, metrics, outputs, detail, aggregate, sanitize_model_output


class EvaluationTests(unittest.TestCase):
    def test_zero_area_boundary_box_filtered_but_invalid_boxes_rejected(self):
        out, n = sanitize_model_output([[0, 0, 0, 10, .01, 0], [0, 0, 10, 10, .8, 0]])
        self.assertEqual(n, 1)
        self.assertEqual(len(out), 1)
        with self.assertRaises(AssertionError):
            sanitize_model_output([[10, 0, 0, 10, .1, 0]])

    def test_threshold_prefix_reproduces_full_rematching(self):
        gt = np.array([[0, 0, 10, 10, 0], [1, 0, 11, 10, 0], [20, 20, 30, 30, 1]], float)
        pred = np.array([[0, 0, 10, 10, .9, 0], [0, 0, 10, 10, .8, 0],
                         [20, 20, 30, 30, .7, 0], [20, 20, 30, 30, .7, 1],
                         [40, 40, 50, 50, .2, 1]], float)
        pairs, _, fp = match(gt, pred)
        for t in [.05, .2, .7, .71, .8, .9, .95]:
            actual, _, unmatched = match(gt, finish(pred, t))
            self.assertEqual(len(actual), sum(pred[i, 4] >= t for i, _, _ in pairs))
            self.assertEqual(len(unmatched), sum(pred[i, 4] >= t for i in fp))

    def test_selection_respects_budget_and_tie_breaks(self):
        points = [dict(threshold=t, **metrics(tp, fp, 100)) for t, tp, fp in
                  [(.1, 99, 11), (.2, 90, 10), (.3, 90, 8), (.4, 90, 8), (.5, 80, 2)]]
        self.assertEqual(choose(points, 10)['threshold'], .4)
        with self.assertRaises(ValueError): choose(points, 1)

    def test_wbf_keeps_low_score_singleton_before_final_threshold(self):
        p = {'fusion_iou': .55, 'input_confidence': .01, 'fusion_output_floor': .005, 'max_det': 300}
        out = outputs([[0, 0, 10, 10, .4, 0]], [], 100, 100, p)
        self.assertAlmostEqual(out['ensemble_wbf'][0, 4], .2)
        self.assertEqual(len(finish(out['ensemble_wbf'], .19)), 1)
        self.assertEqual(len(finish(out['ensemble_wbf'], .21)), 0)

    def test_other_class_cannot_match_and_counts_conserve(self):
        gt = np.array([[0, 0, 10, 10, 0], [30, 30, 70, 70, 1]], float)
        pred = np.array([[0, 0, 10, 10, .9, 1], [30, 30, 70, 70, .8, 1]], float)
        d = detail(gt, pred, 640, 640)
        self.assertEqual(d['overall']['tp'], 1)
        self.assertEqual(d['overall']['fp'], 1)
        self.assertEqual(d['overall']['fn'], 1)
        a = aggregate([d, d])
        self.assertEqual(a['overall']['gt'], 4)
        self.assertEqual(sum(x['tp'] for x in a['classes'].values()), 2)
        self.assertEqual(sum(x['gt'] for x in a['sizes'].values()), 4)


if __name__ == '__main__': unittest.main()
