import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from models.trajectory_dictionary import TrajectoryDictionary
from models.scripts import build_trajectory_dictionary as builder


class Stroke(torch.nn.Module):
    max_strokes = 2
    def generate(self, images, max_strokes=None, threshold=.5):
        return [dict(masks=image.new_ones(1, 256, 256), starts=image.new_tensor([[.2, .3, .05]]),
                     boxes=image.new_tensor([[0., 0., 1., 1.]]), terminated=True,
                     truncated=False, stop_reason='invalid') for image in images]


class Point(torch.nn.Module):
    max_points = 4
    def generate(self, images, first_points, max_points=None, threshold=.5):
        return [dict(points=torch.stack([p, p+p.new_tensor([.1, .1, 0.])]),
                     stop_reason='invalid', terminated=True, truncated=False) for p in first_points]


class TrajectoryDictionaryTests(unittest.TestCase):
    def test_build_lookup_variants_portability_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'source'; output = root/'library'
            for character, variants in [('你', ['a', 'b']), ('好', ['a'])]:
                (source/character).mkdir(parents=True)
                for variant in variants:
                    image = np.full((100, 80), 255, np.uint8); image[20:80, 30:40] = 0
                    Image.fromarray(image).save(source/character/(variant+'.gif'))
            for name in ['stroke.pt', 'point.pt']:
                (root/name).write_bytes(b'fixture; actual loaders replaced')
            command = ['builder', '--source', str(source), '--output', str(output), '--device', 'cpu',
                       '--stroke-checkpoint', str(root/'stroke.pt'), '--point-checkpoint', str(root/'point.pt'),
                       '--warmup', '1', '--threads', '1']
            config = dict(config=dict(validity_threshold=.5))
            with patch.object(sys, 'argv', command), patch.object(builder, 'load_stroke', return_value=(Stroke().eval(), config)), \
                    patch.object(builder, 'load_point', return_value=(Point().eval(), config)), contextlib.redirect_stdout(io.StringIO()):
                builder.main()
            library = TrajectoryDictionary(output)
            self.assertEqual(library.available_characters(), ['你', '好'])
            self.assertEqual(len(library.index['characters']['你']['variants']), 2)
            self.assertEqual(library.lookup('你')['variant_id'], 'a')
            self.assertEqual(library.lookup('你', variant='b')['variant_id'], 'b')
            normalized = library.lookup('你')['strokes'][0][0]
            np.testing.assert_allclose(normalized, [.2, .3, .05], atol=1e-6)
            pixel = library.lookup('你', normalized=False)['strokes'][0][0]
            np.testing.assert_allclose(pixel, [.2*255, .3*255, .05*256], atol=1e-5)
            self.assertEqual(len(library.lookup_text('你好你')['characters']), 3)
            self.assertFalse(library.lookup_text('你好')['contains_robot_motion_commands'])
            with self.assertRaises(KeyError): library.lookup('陌')
            with self.assertRaises(ValueError): library.lookup('你好')
            self.assertTrue((output/'characters/你/a.html').is_file())
            self.assertFalse((output/'characters/你/visualizations').exists())
            # All artifact paths are library-relative; moving the whole directory works.
            moved = root/'moved'; output.rename(moved)
            self.assertEqual(TrajectoryDictionary(moved).lookup('好')['character'], '好')
            resume = command.copy(); resume[resume.index('--output')+1] = str(moved); resume.append('--resume')
            with patch.object(sys, 'argv', resume), patch.object(builder, 'load_stroke') as loader, contextlib.redirect_stdout(io.StringIO()):
                builder.main()
                loader.assert_not_called()
            self.assertEqual(json.loads((moved/'summary.json').read_text())['status'], 'complete')

    def test_truncated_default_and_invalid_offsets_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = dict(schema_version=1, characters={'一': dict(default_variant=None, variants=[dict(
                id='a', status='needs_review', trajectory_json='a.json')])})
            (root/'dictionary.json').write_text(json.dumps(index))
            data = dict(character='一', variant_id='a', points_xyw_px=[[1., 2., 3.]], stroke_offsets=[0, 1])
            (root/'a.json').write_text(json.dumps(data))
            library = TrajectoryDictionary(root)
            with self.assertRaises(ValueError): library.lookup('一')
            with self.assertRaises(ValueError): library.lookup('一', variant='a')
            self.assertEqual(len(library.lookup('一', variant='a', allow_incomplete=True)['strokes']), 1)
            data['stroke_offsets'] = [0, 2]
            (root/'a.json').write_text(json.dumps(data))
            with self.assertRaises(ValueError): library.lookup('一', variant='a', allow_incomplete=True)


if __name__ == '__main__':
    unittest.main()
