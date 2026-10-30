import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from models.datasets.casia_pot import decode_label, iter_pot, normalized_strokes
from models.training.casia_metrics import dtw_xy, FeatureMoments


def pot_record(char, strokes):
    tag = ord(char) << 8 if char.isascii() else int.from_bytes(char.encode('gbk'), 'big')
    body = b''
    for stroke in strokes:
        for x, y in stroke:
            body += struct.pack('<hh', x, y)
        body += struct.pack('<hh', -1, 0)
    body += struct.pack('<hh', -1, -1)
    return struct.pack('<HIH', len(body)+8, tag, len(strokes))+body


class CasiaEvaluationTests(unittest.TestCase):
    def test_download_extraction_repeat_and_no_overwrite(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            content = pot_record('啊', [[(0, 0), (1, 1)]])
            with zipfile.ZipFile(p/'Pot1.1Test.zip', 'w') as archive:
                archive.writestr('writer.pot', content)
            command = [sys.executable, str(root/'models/scripts/download_casia_olhwdb.py'),
                       '--output', str(p), '--extract-only']
            for _ in range(2):
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads((p/'download_1.1_test.json').read_text())
            self.assertEqual(report['existing_verified'], 1)
            extracted = p/'1.1/test/writer.pot'
            extracted.write_bytes(b'keep this differing file')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(extracted.read_bytes(), b'keep this differing file')

    def test_official_label_byte_order_and_stroke_markers(self):
        self.assertEqual(decode_label(0xb0a1), '啊')
        self.assertEqual(decode_label(0x41), 'A')
        self.assertEqual(decode_label(0x4100), 'A')
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'writer.pot'
            record = pot_record('啊', [[(10, 20), (20, 20)], [(15, 10), (15, 30)]])
            self.assertEqual(record[2:6], b'\xa1\xb0\0\0')
            p.write_bytes(record)
            sample = next(iter_pot(p))
            self.assertEqual(sample['char'], '啊')
            self.assertEqual(len(sample['strokes']), 2)
            strokes, transform = normalized_strokes(sample['strokes'])
            points = np.concatenate(strokes)
            np.testing.assert_allclose((points.min(0)+points.max(0))/2, [127.5, 127.5])
            self.assertEqual(float(np.ptp(points[:, 1])), 239)
            self.assertGreater(transform['scale'], 0)
            p.write_bytes(record[:-1])
            with self.assertRaisesRegex(ValueError, 'truncated'):
                list(iter_pot(p))

    def test_exact_dtw_order_and_cost(self):
        a = np.array([[0., 0.], [1., 0.], [2., 0.]])
        self.assertEqual(dtw_xy(a, a)['dtw_sum_px'], 0)
        shifted = dtw_xy(a, a+[0, 1])
        self.assertEqual(shifted['dtw_sum_px'], 3)
        self.assertEqual(shifted['dtw_path_mean_px'], 1)
        self.assertGreater(dtw_xy(a, a[::-1])['dtw_sum_px'], 0)
        self.assertIsNone(dtw_xy(a, np.empty((0, 2)))['dtw_sum_px'])

    def test_streaming_fid_moments_match_full_covariance(self):
        x = np.random.RandomState(42).normal(size=(17, 4))
        moments = FeatureMoments()
        for batch in [x[:2], x[2:13], x[13:]]:
            moments.update(batch)
        np.testing.assert_allclose(moments.mean, x.mean(0), atol=1e-14)
        np.testing.assert_allclose(moments.covariance(), np.cov(x, rowvar=False), atol=1e-14)
        try:
            from pytorch_fid.fid_score import calculate_frechet_distance
        except ImportError:
            return
        self.assertAlmostEqual(calculate_frechet_distance(moments.mean, moments.covariance(),
                                                        moments.mean, moments.covariance()), 0., places=8)

    def test_image_only_cli_and_resume(self):
        from models.networks.single_stroke import SingleStrokeModel, MODE as POINT_MODE
        from models.networks.ordered_stroke import OrderedStrokeModel, MODE
        torch.set_num_threads(2)
        stroke_config = dict(dim=16, heads=4, encoder_layers=1, decoder_layers=2,
                             max_strokes=2, character_backbone='resnet18', dropout=0.)
        point_config = dict(dim=16, heads=4, encoder_layers=1, decoder_layers=4, max_points=3, dropout=0.)
        stroke, point = OrderedStrokeModel(**stroke_config), SingleStrokeModel(**point_config)
        with torch.no_grad():
            stroke.valid_head.weight.zero_(); stroke.valid_head.bias.fill_(20)
            point.valid_head.weight.zero_(); point.valid_head.bias.fill_(-20)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p/'pots').mkdir()
            (p/'pots/writer.pot').write_bytes(b''.join(pot_record(char, [[(0, 0), (10, 10)], [(0, 10), (10, 0)]])
                                                       for char in ['啊', '一', 'A']))
            for name, model, mode, config in [('stroke', stroke, MODE, stroke_config), ('point', point, POINT_MODE, point_config)]:
                torch.save(dict(training_mode=mode, model=model.state_dict(), epoch=1, step=1,
                                config=dict(model=config, validity_threshold=.5)), p/(name+'.pt'))
            root = Path(__file__).resolve().parents[2]
            command = [sys.executable, str(root/'models/scripts/evaluate_casia_olhwdb.py'),
                       '--pot-root', str(p/'pots'), '--output', str(p/'out'), '--device', 'cpu',
                       '--stroke-checkpoint', str(p/'stroke.pt'), '--point-checkpoint', str(p/'point.pt'),
                       '--threads', '2', '--warmup', '1', '--visualize-count', '1', '--skip-fid']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            summary = json.loads((p/'out/summary.json').read_text())
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['metrics']['images'], 2)
            self.assertEqual(summary['metrics']['dtw_valid_images'], 2)
            self.assertEqual(summary['sessions'][0]['warmup_runs'], 1)
            self.assertEqual(summary['fid']['status'], 'explicitly_skipped')
            rows = [json.loads(r) for r in (p/'out/per_image.jsonl').read_text().splitlines()]
            self.assertEqual([r['char'] for r in rows], ['啊', '一'])
            with np.load(p/'out'/rows[0]['artifact']) as z:
                self.assertEqual(z['gt_points'].shape, (4, 2))
                self.assertEqual(z['points'].shape, (2, 3))
            example = p/'out'/rows[0]['example_directory']
            self.assertTrue((example/'directions.png').is_file())
            self.assertFalse((example/'visualizations').exists())
            result = subprocess.run(command+['--resume'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            resumed = json.loads((p/'out/summary.json').read_text())
            self.assertEqual(resumed, summary)
            self.assertEqual(len((p/'out/per_image.jsonl').read_text().splitlines()), 2)


if __name__ == '__main__':
    unittest.main()
