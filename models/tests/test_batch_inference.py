import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch

from models.scripts.infer_two_stage_batch import reconstruction_metrics, summarize, prepare_input


class BatchInferenceTests(unittest.TestCase):
    def test_binarize_preserves_canvas_and_gray_ink(self):
        image = np.full((256, 256), 255, np.uint8)
        image[20:100, 50:60] = 175
        binary, threshold = prepare_input(image, True)
        self.assertEqual(binary.shape, image.shape)
        self.assertEqual(set(np.unique(binary).tolist()), {0, 255})
        self.assertTrue((binary[20:100, 50:60] == 0).all())
        self.assertIsNotNone(threshold)
        original, threshold = prepare_input(image)
        np.testing.assert_array_equal(original, image)
        self.assertIsNone(threshold)

    def test_overlap_metrics_and_empty_conventions(self):
        foreground = np.array([[True, True], [False, False]])
        predicted = np.array([[True, False], [True, False]])
        metrics = reconstruction_metrics(foreground, predicted)
        self.assertAlmostEqual(metrics['iou'], 1/3)
        for key in ['dice', 'precision', 'recall']:
            self.assertEqual(metrics[key], .5)
        empty = np.zeros_like(foreground)
        self.assertEqual(reconstruction_metrics(foreground, empty)['precision'], 0)
        self.assertEqual(reconstruction_metrics(empty, empty)['iou'], 1)

    def test_macro_average_and_cap_denominators(self):
        rows = [dict(status='ok', inference_seconds=t, stroke_count=n, point_count=n*2,
                     reconstruction=dict(iou=i, dice=i, precision=i, recall=i),
                     stroke_truncated=cap, point_truncations=points,
                     stroke_stop_reason='stroke_cap' if cap else 'invalid',
                     point_stop_reasons=['point_cap']*points+['invalid']*(n-points))
                for t, n, i, cap, points in [(1., 1, .2, False, 1), (3., 3, .8, True, 0)]]
        report = summarize(rows+[dict(status='failed')])
        self.assertEqual(report['reconstruction_mean']['iou'], .5)
        self.assertEqual(report['timing_seconds']['mean'], 2)
        self.assertEqual(report['point_cap_stroke_rate'], .25)
        self.assertEqual(report['point_cap_image_rate'], .5)
        self.assertEqual(report['any_cap_image_rate'], 1)
        self.assertEqual(report['failed_images'], 1)

    def test_cli_warmup_resume_and_saved_metrics(self):
        from models.networks.single_stroke import SingleStrokeModel, MODE as POINT_MODE
        from models.networks.ordered_stroke import OrderedStrokeModel, MODE
        torch.set_num_threads(2)
        stroke_settings = dict(dim=16, heads=4, encoder_layers=1, decoder_layers=2,
                               max_strokes=2, character_backbone='resnet18', dropout=0.)
        point_settings = dict(dim=16, heads=4, encoder_layers=1, decoder_layers=4, max_points=3, dropout=0.)
        stroke = OrderedStrokeModel(**stroke_settings)
        point = SingleStrokeModel(**point_settings)
        with torch.no_grad():
            stroke.valid_head.weight.zero_(); stroke.valid_head.bias.fill_(20)
            point.valid_head.weight.zero_(); point.valid_head.bias.fill_(-20)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name, model, mode, settings in [('stroke', stroke, MODE, stroke_settings),
                                                 ('point', point, POINT_MODE, point_settings)]:
                torch.save(dict(training_mode=mode, model=model.state_dict(), epoch=1, step=1,
                                config=dict(model=settings, validity_threshold=.5)), tmp/(name+'.pt'))
            with (tmp/'metadata.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['image_relative_path','source_category','script_type','char','calligrapher'])
                writer.writeheader()
                for i in range(2):
                    image = np.full((256, 256), 255, np.uint8)
                    image[30:100, 50+i*10:70+i*10] = 175
                    cv2.imwrite(str(tmp/f'image{i}.png'), image)
                    writer.writerow(dict(image_relative_path=f'image{i}.png', source_category='Tie',
                                         script_type='楷书', char='一', calligrapher='fixture'))
            root = Path(__file__).resolve().parents[2]
            command = [sys.executable, str(root/'models/scripts/infer_two_stage_batch.py'),
                       '--metadata', str(tmp/'metadata.csv'), '--image-root', str(tmp),
                       '--stroke-checkpoint', str(tmp/'stroke.pt'), '--point-checkpoint', str(tmp/'point.pt'),
                       '--output', str(tmp/'output'), '--device', 'cpu', '--threads', '2', '--warmup', '1', '--binarize']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            summary_path = tmp/'output/summary.json'
            report = json.loads(summary_path.read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(report['per_image']), 2)
            self.assertEqual(report['sessions'][0]['warmup_runs'], 1)
            self.assertEqual(report['metrics']['total_strokes'], 4)
            self.assertEqual(report['metrics']['stroke_cap_image_rate'], 1)
            self.assertAlmostEqual(report['metrics']['timing_seconds']['total'],
                                   sum(r['inference_seconds'] for r in report['per_image']))
            for row in report['per_image']:
                directory = tmp/'output'/row['prediction_dir']
                reconstruction = cv2.imread(str(directory/'reconstruction.png'), 0) == 0
                original = cv2.imread(row['image'], 0)
                _, binary = cv2.threshold(original, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                self.assertEqual(row['reconstruction'], reconstruction_metrics(binary == 0, reconstruction))
                np.testing.assert_array_equal(cv2.imread(row['input_image'], 0), binary)
                self.assertTrue(row['input_binarized'])
                self.assertFalse((directory/'visualizations').exists())
                for name in ['directions.png', 'overview.png', 'strokes.png']:
                    self.assertTrue((directory/name).is_file())
                with np.load(directory/'prediction.npz') as z:
                    self.assertEqual(z['masks'].dtype, np.bool_)
                    self.assertEqual(z['points'].shape, (2, 3))
            result = subprocess.run(command+['--resume'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(json.loads(summary_path.read_text()), report)


if __name__ == '__main__':
    unittest.main()
