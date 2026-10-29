"""Synthetic cases for area ignore, empty sizes, score matching and class errors."""
import copy
import unittest
from evaluate import coco_ap, detail, merge_details

def dataset(boxes, classes=None):
    classes=classes or [0]*len(boxes)
    return {'info':{},'images':[{'id':1,'width':400,'height':400,'file_name':'synthetic.png'}],
            'annotations':[{'id':i+1,'image_id':1,'category_id':c,'bbox':b,'area':b[2]*b[3],'iscrowd':0} for i,(b,c) in enumerate(zip(boxes,classes))],
            'categories':[{'id':0,'name':'hat'},{'id':1,'name':'person'}]}

def prediction(box,cls=0,score=.9):return {'image_id':1,'category_id':cls,'bbox':box,'score':score}

class EvaluationFixtures(unittest.TestCase):
    def test_perfect_all_three_sizes(self):
        boxes=[[0,0,10,10],[20,20,40,40],[100,100,100,100]]
        result=coco_ap(dataset(boxes),[prediction(b) for b in boxes])
        for name in ['all','small','medium','large']:
            self.assertAlmostEqual(result[name]['AP50_95'],1.)

    def test_wrong_class_is_not_true_positive(self):
        r=coco_ap(dataset([[0,0,10,10]]),[prediction([0,0,10,10],1)])
        self.assertEqual(r['small']['AP50_95'],0.)

    def test_size_without_ground_truth_is_undefined(self):
        r=coco_ap(dataset([[0,0,10,10]]),[prediction([0,0,10,10])])
        self.assertIsNone(r['large']['AP50_95'])

    def test_high_score_false_positive_reduces_ap(self):
        r=coco_ap(dataset([[0,0,10,10]]),[prediction([50,50,10,10],score=.99),prediction([0,0,10,10],score=.9)])
        self.assertAlmostEqual(r['small']['AP50_95'],.5)

    def test_out_of_area_matched_large_box_is_ignored_for_small_ap(self):
        boxes=[[0,0,10,10],[100,100,100,100]]
        r=coco_ap(dataset(boxes),[prediction(boxes[1],score=.99),prediction(boxes[0],score=.9)])
        self.assertAlmostEqual(r['small']['AP50_95'],1.)

    def test_no_predictions(self):
        r=coco_ap(dataset([[0,0,10,10]]),[])
        self.assertEqual(r['small']['AP50_95'],0.)

    def test_recall_size_uses_gt_after_global_matching(self):
        gt=[[0,0,10,10,0],[30,30,130,130,0]]
        d=detail(gt,[[30,30,130,130,.9,0]],400,400)
        self.assertEqual(d['area']['small']['all']['tp'],0)
        self.assertEqual(d['area']['large']['all']['tp'],1)
        self.assertEqual(d['overall']['fp'],0)

    def test_duplicate_prediction_is_false_positive(self):
        d=detail([[0,0,10,10,0]],[[0,0,10,10,.9,0],[0,0,10,10,.8,0]],400,400)
        self.assertEqual((d['overall']['tp'],d['overall']['fp'],d['overall']['fn']),(1,1,0))

    def test_wrong_class_counts_false_positive_and_false_negative(self):
        d=detail([[0,0,10,10,0]],[[0,0,10,10,.9,1]],400,400)
        self.assertEqual((d['overall']['tp'],d['overall']['fp'],d['overall']['fn']),(0,1,1))

    def test_area_partition_and_micro_aggregation(self):
        gt=[[0,0,32,32,0],[40,40,136,136,1]]
        d=detail(gt,[[0,0,32,32,.9,0]],400,400)
        self.assertEqual(d['area']['small']['all']['gt'],0)
        self.assertEqual(d['area']['medium']['all']['gt'],1)
        self.assertEqual(d['area']['large']['all']['gt'],1)
        total=merge_details([d,d])['overall']
        self.assertEqual(total['gt'],4);self.assertEqual(total['tp'],2)
        self.assertAlmostEqual(total['recall'],.5)

if __name__=='__main__':unittest.main(verbosity=2)
