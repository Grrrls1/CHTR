import unittest
import numpy as np
import torch

from models.training.two_stage_metrics import compare_trajectories, summarize


def prediction(path):
    return dict(points=torch.from_numpy(path/np.array([255,255,256],np.float32)),terminated=True,truncated=False)


class TwoStageMetricTests(unittest.TestCase):
    def test_perfect_missing_and_extra_strokes_are_distinguished(self):
        path=np.array([[30,40,4],[60,70,6]],np.float32)
        perfect,_=compare_trajectories([path],[prediction(path)])
        self.assertAlmostEqual(perfect['character_trajectory_iou'],1.)
        self.assertAlmostEqual(perfect['aligned_xy_mae_px'],0.,places=5)
        missing,_=compare_trajectories([path],[])
        self.assertEqual(missing['ordered_trajectory_iou'],0.)
        self.assertIsNone(missing['aligned_xy_mae_px'])
        self.assertEqual(missing['missing_strokes'],1)
        extra,_=compare_trajectories([path],[prediction(path),prediction(path)])
        self.assertEqual(extra['ordered_trajectory_iou'],.5)
        self.assertEqual(extra['extra_strokes'],1)
        report=summarize([perfect,missing,extra])
        self.assertEqual(report['predicted_strokes'],3)
        self.assertEqual(report['missing_strokes'],1)
        self.assertEqual(report['extra_strokes'],1)
        self.assertAlmostEqual(report['ordered_trajectory_iou'],.5)

    def test_first_point_error_is_included_and_empty_rates_are_null(self):
        gt=np.array([[30,40,4]],np.float32)
        pred=np.array([[40,40,6]],np.float32)
        row,_=compare_trajectories([gt],[prediction(pred)])
        self.assertAlmostEqual(row['aligned_xy_mae_px'],5.,places=5)
        self.assertAlmostEqual(row['aligned_width_mae_px'],2.,places=5)
        empty,_=compare_trajectories([gt],[])
        report=summarize([empty])
        self.assertIsNone(report['terminated_rate'])
        self.assertIsNone(report['aligned_endpoint_l2_px'])


if __name__=='__main__':
    unittest.main()
