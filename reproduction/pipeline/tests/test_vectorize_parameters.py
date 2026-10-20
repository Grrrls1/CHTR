import sys
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vectorize_strokes_batch import resolve_start_cursor


class VectorizeParameterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        mask = np.zeros((256, 256), np.uint8)
        cv2.line(mask, (30, 40), (220, 200), 1, 9)
        self.input = self.root / 'stroke.png'
        cv2.imwrite(str(self.input), 255 - mask * 255)
        self.row = dict(stroke_id='fixture_stroke000', input_png=str(self.input),
                        character='一', stroke_index=0)

    def tearDown(self):
        self.temp.cleanup()

    def test_random_cursor_is_unchanged_and_deterministic_modes_are_normalized(self):
        initial = np.asarray([[[[.25, .75]]]], np.float32)
        cursor, detail = resolve_start_cursor('random', initial, self.row)
        np.testing.assert_array_equal(cursor, initial)
        self.assertEqual(detail['status'], 'loader_random')
        cursor, detail = resolve_start_cursor('skeleton', initial, self.row)
        self.assertEqual(cursor.shape, initial.shape)
        self.assertTrue(np.all(cursor >= 0) and np.all(cursor < 1))
        self.assertEqual(detail['source'], 'skeleton')

    def test_mmah_cursor_requires_and_uses_ordered_median(self):
        source = self.root / 'mmah.npz'
        np.savez(source, median_000=np.asarray([[220, 200], [30, 40]], np.float32))
        refs = {'一': dict(source_npz_path=str(source))}
        cursor, detail = resolve_start_cursor('mmah', np.zeros((1, 1, 2), np.float32), self.row, refs)
        self.assertGreater(cursor[0, 0, 0], .7)
        self.assertGreater(cursor[0, 0, 1], .6)
        self.assertEqual(detail['status'], 'mmah_median_endpoint')

        with self.assertRaises(ValueError):
            resolve_start_cursor('mmah', np.zeros((1, 1, 2), np.float32), self.row, {})


if __name__ == '__main__':
    unittest.main()
