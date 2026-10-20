import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_mixed_sdnet_dataset import annotation_has_mask, augment_masks, save_pair


class MixedDatasetTests(unittest.TestCase):
    def test_empty_uncompressed_rle_is_rejected(self):
        self.assertFalse(annotation_has_mask(
            {'segmentation': {'counts': [14400], 'size': [120, 120]}}, 120, 120))
        self.assertTrue(annotation_has_mask(
            {'segmentation': {'counts': [10, 4, 14386], 'size': [120, 120]}}, 120, 120))

    def test_augmentation_is_deterministic_and_keeps_strokes(self):
        masks = np.zeros((2, 256, 256), np.uint8)
        masks[0, 60:80, 80:150] = 1
        masks[1, 100:180, 120:140] = 1
        first = augment_masks(masks, '永', 1)
        second = augment_masks(masks, '永', 1)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(first.reshape(2, -1).sum(1) > 0))
        self.assertFalse(np.array_equal(first, masks))

    def test_saved_pair_has_training_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / 'reference.npz'
            destination = root / 'pair.npz'
            masks = np.zeros((1, 256, 256), np.uint8)
            masks[0, 10:30, 20:50] = 1
            np.savez_compressed(reference_path,
                reference_single_image=masks, stroke_label=np.array([1]),
                reference_single_centroid=np.array([[34.5, 19.5]], np.float32),
                reference_color_image=np.zeros((3, 256, 256), np.float32))
            save_pair(destination, '一', dict(reference_path=str(reference_path)), masks, 'none')
            with np.load(destination, allow_pickle=False) as z:
                required = {'reference_single_image', 'reference_single_centroid',
                            'reference_color_image', 'stroke_label', 'target_image',
                            'target_single_image'}
                self.assertTrue(required.issubset(z.files))
                self.assertTrue(np.array_equal(z['target_image'][0], masks[0]))


if __name__ == '__main__':
    unittest.main()
