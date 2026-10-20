"""Check reference-bank conversion preserves order and does not invent target masks."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_mmah_reference_bank import build_one
from common import digest


class ReferenceBankTests(unittest.TestCase):
    def test_build_reference_only_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.npz'
            masks = np.zeros((2, 256, 256), np.uint8)
            masks[0, 10:20, 10:20] = 1
            masks[1, 30:40, 40:50] = 1
            np.savez_compressed(source, character=np.array('永'),
                                stroke_order=np.array([0, 1]), ordered_stroke_masks=masks)
            row = dict(character='永', stroke_count=2, stroke_label24_proposed=[0, 13],
                       ordered_strokes_path=str(source), ordered_strokes_sha256=digest(source))
            output = root / 'bank'
            output.mkdir()
            item, skip = build_one(row, output)
            self.assertIsNone(skip)
            with np.load(item['reference_path'], allow_pickle=False) as z:
                self.assertTrue(np.array_equal(z['reference_single_image'], masks))
                self.assertEqual(z['stroke_label'].tolist(), [0, 13])
                self.assertEqual(z['reference_single_centroid'].shape, (2, 2))
                self.assertNotIn('target_single_image', z.files)

    def test_unresolved_label_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            row = dict(character='永', stroke_count=1, stroke_label24_proposed=[-1])
            item, skip = build_one(row, Path(directory))
            self.assertIsNone(item)
            self.assertEqual(skip['reason'], 'unresolved_category')


if __name__ == '__main__':
    unittest.main()
