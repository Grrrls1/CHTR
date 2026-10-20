import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clean_stroke_masks import clean_one


class CleanStrokeMasksTests(unittest.TestCase):
    def settings(self):
        return Namespace(target_dilation=1, tiny_pixels=16, tiny_fraction=.005,
            substantial_pixels=16, foreign_relative=.25, foreign_overlap=.60,
            minimum_main_prior_support=.30, prior_margin=.20,
            probability_margin=-.05, max_bridge_fraction=.10,
            connection_max_gap=8.0, connection_corridor_support=.80,
            reference_connection_dilation=4, spur_pixels=12,
            spur_max_neighbors=2, spur_iterations=2,
            smooth_close_kernel=5, smooth_open_kernel=3,
            max_smoothing_change_fraction=.05, max_hole_area=256,
            allow_multiple_components=False)

    def test_redundant_disconnected_component_is_removed(self):
        masks = np.zeros((2, 256, 256), bool)
        masks[0, 100:150, 100:120] = True
        masks[0, 50:60, 50:60] = True
        masks[1, 48:62, 48:62] = True
        priors = masks.astype(np.float32)
        target = np.any(masks, axis=0)
        _, cleaned, actions = clean_one(masks, priors, target, None, self.settings())
        self.assertFalse(cleaned[0, 50:60, 50:60].any())
        self.assertGreater(cleaned[0, 100:150, 100:120].sum(), 950)
        reasons = [action['action'] for action in actions[0]['actions']]
        self.assertIn('remove_redundant_foreign_component', reasons)

    def test_large_unexplained_component_is_kept_for_review(self):
        masks = np.zeros((1, 256, 256), bool)
        masks[0, 100:140, 100:120] = True
        masks[0, 30:50, 30:50] = True
        priors = masks.astype(np.float32)
        target = masks[0]
        settings = self.settings(); settings.allow_multiple_components = True
        _, cleaned, actions = clean_one(masks, priors, target, None, settings)
        self.assertGreater(cleaned[0, 30:50, 30:50].sum(), 350)
        self.assertEqual(2, __import__('clean_stroke_masks').component_count(cleaned[0]))
        reasons = [action['action'] for action in actions[0]['actions']]
        self.assertIn('keep_ambiguous', reasons)

    def test_nearby_components_connect_only_in_target_reference_corridor(self):
        masks = np.zeros((1, 256, 256), bool)
        masks[0, 100:111, 90:101] = True
        masks[0, 100:111, 105:116] = True
        target = masks[0].copy(); target[100:111, 100:106] = True
        prior = target[None].astype(np.float32)
        _, cleaned, actions = clean_one(masks, prior, target, None, self.settings())
        self.assertEqual(1, sum(v >= 16 for v in [cleaned[0].sum()]))
        self.assertEqual(1, __import__('clean_stroke_masks').component_count(cleaned[0], 16))
        reasons = [action['action'] for action in actions[0]['actions']]
        self.assertIn('connect_target_reference_corridor', reasons)

    def test_edge_spur_outside_prior_is_pruned(self):
        masks = np.zeros((1, 256, 256), bool)
        masks[0, 100:120, 100:120] = True
        masks[0, 109, 120:124] = True
        prior = np.zeros_like(masks, np.float32); prior[0, 100:120, 100:120] = 1
        target = masks[0]
        _, cleaned, actions = clean_one(masks, prior, target, None, self.settings())
        self.assertLess(cleaned.sum(), masks.sum())
        reasons = [action['action'] for action in actions[0]['actions']]
        self.assertIn('remove_edge_spurs', reasons)

    def test_internal_hole_is_filled_and_final_mask_is_single_component(self):
        masks = np.zeros((1, 256, 256), bool)
        masks[0, 80:140, 80:140] = True
        masks[0, 100:106, 100:106] = False
        masks[0, 30:40, 30:40] = True
        priors = masks.astype(np.float32); target = masks[0]
        _, cleaned, actions = clean_one(masks, priors, target, None, self.settings())
        self.assertTrue(cleaned[0, 102, 102])
        self.assertFalse(cleaned[0, 35, 35])
        self.assertEqual(1, __import__('clean_stroke_masks').component_count(cleaned[0]))
        reasons = [action['action'] for action in actions[0]['actions']]
        self.assertIn('fill_internal_holes', reasons)


if __name__ == '__main__':
    unittest.main()
