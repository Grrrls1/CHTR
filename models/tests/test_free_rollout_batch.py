"""Batched decoding must preserve per-image paths and full training gradients."""
import contextlib
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.networks.model import StrokeTrajectoryModel
from models.networks import free_rollout as free
from models.scripts.train_free_rollout import train_microbatch


CONFIG = dict(geometry_loss=dict(type='pixel_charbonnier', epsilon_px=.5, scale_px=10.),
              alignment_points=8, rollout=dict(max_strokes=3, max_points=6))


class BatchedRolloutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.model = StrokeTrajectoryModel(dim=16, heads=2, stroke_layers=1, point_layers=1,
            dropout=0., memory_size=2, max_strokes=3, max_points=6, separate_branches=True).eval()
        self.images = torch.rand(4, 1, 32, 32)
        self.images[:, 0, 0, 0] = torch.arange(4.)
        self.samples = [dict(image=image, image_size=32,
            points=torch.tensor([[.1,.2,.1,1.,0,0],[.2,.2,.1,0,1,0],
                                 [.5,.4,.2,1,0,0],[.6,.5,.2,0,0,1]]),
            stroke_offsets=torch.tensor([0,2,4]), stroke_types=torch.tensor([1,2]),
            stroke_masks=torch.ones(2,32,32)) for image in self.images]

    @contextlib.contextmanager
    def ragged_events(self):
        # Exercise compaction within a stroke and between strokes, plus both caps.
        encode, next_stroke, point = free.encode_image, free.next_stroke, free.point_step
        def marked_encode(model, images):
            pixels, memory, sizes = encode(model, images)
            memory = memory.clone()
            memory[:, 0, 0] = images[:, 0, 0, 0]
            return pixels, memory, sizes
        def marked_stroke(model, pixels, memory, sizes, history, size):
            z, masks, types, cond, bias = next_stroke(model, pixels, memory, sizes, history, size)
            bias = bias.clone(); bias[:, 0] = len(history)
            return z, masks, types, cond, bias
        def marked_point(model, previous, index, cond, memory, bias, caches):
            geometry, states = point(model, previous, index, cond, memory, bias, caches)
            events = []
            for owner, stroke in zip(memory[:, 0, 0].round().tolist(), bias[:, 0].round().tolist()):
                if owner == 0: event = 2 if index == 1 else 0
                elif owner == 1: event = (1 if stroke == 0 else 2) if index == (2 if stroke == 0 else 0) else 0
                elif owner == 2: event = 0
                else: event = 1 if index == 0 else 0
                events.append(event)
            forced = F.one_hot(torch.tensor(events, device=states.device), 3).to(states.dtype)*8
            return geometry, states*.01 + forced
        with patch.object(free, 'encode_image', marked_encode), patch.object(free, 'next_stroke', marked_stroke), patch.object(free, 'point_step', marked_point):
            yield

    def assert_paths_equal(self, actual, expected):
        for a, b in zip(actual, expected):
            for key in ('completed', 'truncated', 'truncation_reason', 'used_ground_truth'):
                self.assertEqual(a[key], b[key])
            self.assertEqual(len(a['strokes']), len(b['strokes']))
            for sa, sb in zip(a['strokes'], b['strokes']):
                self.assertEqual(sa['events'], sb['events'])
                for key in ('geometry', 'state_logits', 'mask_logits', 'type_logits'):
                    torch.testing.assert_close(sa[key], sb[key], rtol=2e-4, atol=2e-5)

    def test_ragged_batch_matches_single_image_paths(self):
        with self.ragged_events():
            batched = free.rollout_batch(self.model, self.images)
            serial = [free.rollout(self.model, image[None]) for image in self.images]
        self.assert_paths_equal(batched, serial)
        self.assertEqual([r['truncation_reason'] for r in batched],
                         [None, None, 'max_points_reached', 'max_strokes_reached'])
        self.assertEqual([[len(s['geometry']) for s in r['strokes']] for r in batched],
                         [[2], [3,1], [6], [1,1,1]])

    def test_visual_batch_matches_variable_count_single_images(self):
        counts = [1,3,2,1]
        batch = free.visual_predictions_batch(self.model, self.images, counts)
        for image, count, rows in zip(self.images, counts, batch):
            single = free.visual_predictions(self.model, image[None], count)
            self.assertEqual(len(rows), count)
            for actual, expected in zip(rows, single):
                for key in actual:
                    torch.testing.assert_close(actual[key], expected[key], rtol=2e-4, atol=2e-5)

    def test_batch_loss_and_gradients_match_accumulated_independent_images(self):
        serial_model = copy.deepcopy(self.model)
        with self.ragged_events():
            batch_log = train_microbatch(self.model, self.samples, CONFIG, 4)
            serial_logs = [train_microbatch(serial_model, [sample], CONFIG, 4) for sample in self.samples]
        for key, actual in batch_log.items():
            self.assertAlmostEqual(actual, sum(r[key] for r in serial_logs), delta=2e-4)
        compared = 0
        for (name, actual), (_, expected) in zip(self.model.named_parameters(), serial_model.named_parameters()):
            self.assertEqual(actual.grad is None, expected.grad is None, name)
            if actual.grad is not None:
                self.assertTrue(torch.isfinite(actual.grad).all(), name)
                torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-3, atol=2e-4, msg=name)
                compared += 1
        self.assertGreater(compared, 50)

    def test_batch_24_and_singleton_use_actual_batch_axis(self):
        with torch.no_grad():
            self.model.state_head.weight.zero_()
            self.model.state_head.bias.copy_(torch.tensor([-8.,-8.,8.]))
        for count in (1,24):
            images = self.images[:1].expand(count,-1,-1,-1).clone()
            rows = free.rollout_batch(self.model, images)
            self.assertEqual(len(rows), count)
            self.assertTrue(all(r['completed'] for r in rows))


if __name__ == '__main__':
    unittest.main()
