import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import digest, png, render, states
from import_rhsedb_targets import export_record
from pack_rhsedb_exploratory import pack_group
from validate_rhsedb_exploratory import validate


class RHSEDBExploratoryTests(unittest.TestCase):
    def test_export_and_pack_at_explicit_threshold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            points = states(np.array([[20, 20, 8, 0, 0, 0], [80, 20, 8, 0, 0, 0]], np.float32), np.array([0, 2]))
            mask = (render(points, np.array([0, 2])) >= 128).astype(np.uint8)
            source = root / '0.npz'
            np.savez_compressed(source, name=np.array('一'), target_image=mask[None],
                                target_single_image=mask[None], stroke_label=np.array([1]))
            inputs = root / 'inputs'
            inputs.mkdir()
            rows, image = export_record(source, 'test', inputs)
            self.assertEqual(image['annotated_mask_union_iou'], 1.0)
            self.assertEqual(rows[0]['writer_status'], 'unknown_in_local_npz')
            trajectory = root / 'trajectory.npz'
            np.savez_compressed(trajectory, points=points, subpath_offsets=np.array([0, 2]))
            upstream = root / 'upstream.png'
            png(upstream, (1 - mask) * 255)
            row = dict(rows[0], trajectory_path=str(trajectory), trajectory_sha256=digest(trajectory),
                       upstream_render_path=str(upstream), upstream_render_sha256=digest(upstream),
                       qc=dict(qc_status='passed', iou=1., upstream_iou=1., native_replay_iou=1.,
                               threshold=.7, fidelity_threshold=.7))
            output = root / 'output'
            output.mkdir()
            packed = pack_group([row], output, .7)
            self.assertEqual(packed['dataset_split'], 'unassigned')
            self.assertEqual(packed['manual_review_status'], 'not_performed')
            self.assertGreaterEqual(packed['whole_iou'], .7)
            self.assertTrue(validate([packed])['valid'])
            with np.load(packed['npz_path'], allow_pickle=False) as z:
                self.assertEqual(z['points'].shape, (2, 6))
                np.testing.assert_array_equal(z['stroke_offsets'], [0, 2])
            bad = dict(row, qc=dict(row['qc'], iou=.69))
            with self.assertRaisesRegex(ValueError, 'below'):
                pack_group([bad], root / 'bad', .7)
            wrong = dict(row, qc=dict(row['qc'], threshold=.95))
            with self.assertRaisesRegex(ValueError, 'wrong experimental threshold'):
                pack_group([wrong], root / 'wrong', .7)


if __name__ == '__main__':
    unittest.main()
