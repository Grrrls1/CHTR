import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import render, validate_points
from skeleton_path_trajectory import build_trajectory, skeletonize_mask


class SinglePathTests(unittest.TestCase):
    def test_thinning_and_single_subpath(self):
        mask = np.zeros((64, 64), np.uint8)
        cv2.line(mask, (8, 16), (56, 48), 1, 9)
        points, offsets, skeleton, diagnostics = build_trajectory(mask.astype(bool), spacing=2.0)
        self.assertEqual(offsets.tolist(), [0, len(points)])
        self.assertGreater(len(skeleton.nonzero()[0]), 0)
        validate_points(points, offsets, size=64, bounds=True)
        self.assertGreater((render(points, offsets, size=64) > 0).sum(), 0)
        self.assertGreater(diagnostics['path_length'], 40)

    def test_median_orients_path(self):
        mask = np.zeros((64, 64), np.uint8)
        cv2.line(mask, (8, 32), (56, 32), 1, 7)
        median = np.asarray([[56, 32], [8, 32]], np.float32)
        points, offsets, _, diagnostics = build_trajectory(mask.astype(bool), median=median, spacing=3.0)
        self.assertGreater(points[0, 0], points[-1, 0])
        self.assertEqual(diagnostics['direction_status'], 'mmah_median')


if __name__ == '__main__':
    unittest.main()
