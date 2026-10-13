from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from review_legacy_test import geometry, image_features
from workflow import sha256


class LegacyScreeningTests(unittest.TestCase):
    def test_identical_decoded_pixels_detect_different_file_encodings(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.png',Path(tmp)/'b.png'
            data=np.random.default_rng(3).integers(0,256,(100,120,3),dtype=np.uint8)
            Image.fromarray(data).save(a,compress_level=0)
            Image.fromarray(data).save(b,compress_level=9)
            orb=cv2.ORB_create(nfeatures=600,edgeThreshold=15,fastThreshold=10)
            fa,fb=image_features(a,orb),image_features(b,orb)
            self.assertNotEqual(sha256(a),sha256(b))
            self.assertEqual(fa[0],fb[0])
            self.assertEqual(fa[1:4],fb[1:4])

    def test_geometric_relation_detects_same_scene_shift(self):
        random=np.random.default_rng(7)
        gray=random.integers(0,256,(480,640),dtype=np.uint8)
        shifted=cv2.warpAffine(gray,np.float32([[1,0,12],[0,1,8]]),(640,480))
        orb=cv2.ORB_create(nfeatures=600,edgeThreshold=15,fastThreshold=10)
        def features(image):
            kp,desc=orb.detectAndCompute(image,None)
            return np.asarray([k.pt for k in kp],dtype=np.float32),desc,image.shape
        relation=geometry(features(gray),features(shifted))
        self.assertIsNotNone(relation)
        self.assertGreater(relation['inliers'],20)

    def test_unrelated_textures_do_not_pass_geometry(self):
        random=np.random.default_rng(9)
        orb=cv2.ORB_create(nfeatures=600,edgeThreshold=15,fastThreshold=10)
        def features():
            image=random.integers(0,256,(480,640),dtype=np.uint8)
            kp,desc=orb.detectAndCompute(image,None)
            return np.asarray([k.pt for k in kp],dtype=np.float32),desc,image.shape
        self.assertIsNone(geometry(features(),features()))

    def test_featureless_images_cannot_claim_scene_match(self):
        self.assertIsNone(geometry((np.empty((0,2)),None,(100,100)),
                                   (np.empty((0,2)),None,(100,100))))


if __name__=='__main__':
    unittest.main()
