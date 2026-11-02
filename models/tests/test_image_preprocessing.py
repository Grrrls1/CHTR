import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image
import torch

from models.scripts.infer_image import preprocess_image, source_xy


class ImagePreprocessingTests(unittest.TestCase):
    def test_aspect_padding_and_inverse_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'input.png'
            a = np.full((100, 200), 255, np.uint8)
            a[20:80, 30:150] = 75
            Image.fromarray(a).save(p)
            image, report = preprocess_image(p)
            self.assertEqual(report['resized_size'], [240, 120])
            self.assertEqual(report['padding'], [8, 68])
            self.assertEqual(image.shape, (256, 256))
            self.assertTrue((image[:68] == 255).all())
            self.assertTrue((image[:, :8] == 255).all())
            self.assertTrue(((image > 0) & (image < 255)).any())
            xy = np.array([[0., 0.], [199., 99.], [30., 20.]])
            matrix = np.array(report['source_to_model'])
            normalized = np.column_stack([xy, np.ones(3)]) @ matrix.T
            np.testing.assert_allclose(source_xy(normalized, report), xy, atol=1e-10)

    def test_crop_and_pad_only_keep_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'input.png'
            a = np.arange(120*160, dtype=np.uint8).reshape(120, 160)
            Image.fromarray(a).save(p)
            image, report = preprocess_image(p, crop=[20, 30, 100, 60], resize='pad-only')
            np.testing.assert_array_equal(image[98:158, 78:178], a[30:90, 20:120])
            np.testing.assert_allclose(source_xy(np.array([[78., 98.], [177., 157.]]), report),
                                       [[20., 30.], [119., 89.]])
            with self.assertRaises(ValueError):
                preprocess_image(p, crop=[150, 0, 30, 40])

    def test_transparency_white_and_optional_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'input.png'
            rgba = np.zeros((80, 80, 4), np.uint8)
            rgba[20:60, 30:50, :3] = 170
            rgba[20:60, 30:50, 3] = 255
            Image.fromarray(rgba).save(p)
            image, report = preprocess_image(p, binarize='otsu')
            self.assertEqual(set(np.unique(image).tolist()), {0, 255})
            self.assertEqual(int(image[20, 20]), 255)
            self.assertEqual(report['binarization'], 'otsu')

    def test_existing_cli_wrapper_and_coordinate_artifact(self):
        from models.networks.single_stroke import SingleStrokeModel, MODE as POINT_MODE
        from models.networks.ordered_stroke import OrderedStrokeModel, MODE
        torch.set_num_threads(2)
        sc = dict(dim=16, heads=4, encoder_layers=1, decoder_layers=2,
                  max_strokes=2, character_backbone='resnet18', dropout=0.)
        pc = dict(dim=16, heads=4, encoder_layers=1, decoder_layers=4, max_points=3, dropout=0.)
        stroke, point = OrderedStrokeModel(**sc), SingleStrokeModel(**pc)
        with torch.no_grad():
            stroke.valid_head.weight.zero_(); stroke.valid_head.bias.fill_(20)
            point.valid_head.weight.zero_(); point.valid_head.bias.fill_(-20)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            for name, model, mode, config in [('stroke', stroke, MODE, sc), ('point', point, POINT_MODE, pc)]:
                torch.save(dict(training_mode=mode, model=model.state_dict(), epoch=1, step=1,
                                config=dict(model=config, validity_threshold=.5)), p/(name+'.pt'))
            Image.fromarray(np.full((100, 200), 255, np.uint8)).save(p/'input.png')
            root = Path(__file__).resolve().parents[2]
            command = [sys.executable, str(root/'models/scripts/infer_image.py'),
                       '--image', str(p/'input.png'), '--output', str(p/'out'),
                       '--stroke-checkpoint', str(p/'stroke.pt'), '--point-checkpoint', str(p/'point.pt'),
                       '--device', 'cpu', '--threads', '2']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads((p/'out/summary.json').read_text())
            self.assertTrue(Path(report['image']).is_file())
            self.assertEqual(report['source_image'], str(p/'input.png'))
            metadata = json.loads((p/'out/preprocess.json').read_text())
            with np.load(p/'out/prediction.npz') as pred, np.load(p/'out/prediction_source_xy.npz') as original:
                np.testing.assert_allclose(original['points_xy'], source_xy(pred['points'], metadata))
                np.testing.assert_array_equal(original['stroke_offsets'], pred['stroke_offsets'])
            self.assertTrue((p/'out/index.html').is_file())
            self.assertFalse((p/'out/visualizations').exists())


if __name__ == '__main__':
    unittest.main()
