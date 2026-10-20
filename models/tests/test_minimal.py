"""Real-data contracts, causal decoding, padding and gradient checks."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from models.datasets.lmdb_dataset import CHTRModelDataset, collate_strokes, normalize_points, denormalize_points
from models.networks.model import StrokeTrajectoryModel
from models.losses.multitask import multitask_loss


class MinimalTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.dataset = CHTRModelDataset(ROOT.parent/'data/chtr_mmah_lmdb_1915')
        # Exercise a small read-only view without copying or modifying the frozen LMDB.
        cls.dataset.ids = cls.dataset.ids[:5]

    @classmethod
    def tearDownClass(cls):
        cls.dataset.close()

    def model(self):
        torch.manual_seed(4)
        return StrokeTrajectoryModel(dim=32, heads=4, stroke_layers=1, point_layers=1, dropout=0, memory_size=4)

    def test_all_records_and_normalization(self):
        for s in self.dataset:
            restored = denormalize_points(s['points'], s['image_size'])
            np.testing.assert_allclose(normalize_points(restored, s['image_size']), s['points'], atol=1e-6)
            ends = np.flatnonzero(s['points'][:, 4:].numpy().sum(-1))+1
            np.testing.assert_array_equal(ends, s['stroke_offsets'][1:])
        s = self.dataset[0]
        self.assertEqual(float(s['stroke_masks'][0, 0, 0]), 0.0)
        self.assertGreater(float(s['stroke_masks'].mean()), 0.01)
        self.assertLess(float(s['stroke_masks'].mean()), 0.2)

    def test_real_batch_backward_and_conditions(self):
        batch = collate_strokes([self.dataset[0], self.dataset[1]])
        model = self.model()
        out = model(batch)
        loss, _ = multitask_loss(out, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name in ['encoder', 'stroke_decoder', 'mask_query', 'type_head', 'last_head',
                     'endpoint_head', 'point_layers', 'geometry_head', 'state_head']:
            grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
            self.assertTrue(grads, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads), name)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0, name)
        # Trajectory loss alone must reach the predicted mask and type conditions.
        model.zero_grad(set_to_none=True)
        out = model(batch)
        out['geometry'].sum().backward()
        for head in (model.mask_query, model.type_head):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in head.parameters()), 0)

    def test_padding_does_not_change_loss(self):
        batch = collate_strokes([self.dataset[0], self.dataset[1]])
        model = self.model().eval()
        with torch.no_grad():
            a = model(batch)
            la, _ = multitask_loss(a, batch)
            padded = copy.deepcopy(batch)
            padded['points'] = F.pad(batch['points'], (0, 0, 0, 3, 0, 1), value=17)
            padded['valid_point'] = F.pad(batch['valid_point'], (0, 3, 0, 1), value=False)
            padded['valid_stroke'] = F.pad(batch['valid_stroke'], (0, 1), value=False)
            padded['masks'] = F.pad(batch['masks'], (0, 0, 0, 0, 0, 1), value=1)
            padded['types'] = F.pad(batch['types'], (0, 1), value=23)
            padded['last'] = F.pad(batch['last'], (0, 1), value=1)
            padded['endpoints'] = F.pad(batch['endpoints'], (0, 0, 0, 1), value=10)
            b = model(padded)
            lb, _ = multitask_loss(b, padded)
        torch.testing.assert_close(la, lb, atol=1e-5, rtol=1e-5)

    def test_no_future_target_leakage(self):
        batch = collate_strokes([self.dataset[0]])
        model = self.model().eval()
        changed = copy.deepcopy(batch)
        changed['points'][:, :, 10:] = 0.8
        with torch.no_grad():
            a, b = model(batch), model(changed)
        # Target point 10 is first consumed as input to predict point 11.
        torch.testing.assert_close(a['geometry'][:, :11], b['geometry'][:, :11])
        torch.testing.assert_close(a['state_logits'][:, :11], b['state_logits'][:, :11])

    def test_affine_augmentation_inverse(self):
        dataset = CHTRModelDataset.__new__(CHTRModelDataset)
        dataset.augment, dataset.scale_range, dataset.translate = True, (0.9, 0.9), 0.0
        source = self.dataset[0]
        points = denormalize_points(source['points'], 256)
        image = (source['image'][0].numpy()*255).astype(np.uint8)
        masks = source['stroke_masks'].numpy()
        augmented, _, transformed_masks, matrix = dataset._augment(points, image, masks, 256)
        inverse = np.linalg.inv(matrix)
        recovered = augmented[:, :2] @ inverse[:2, :2].T + inverse[:2, 2]
        np.testing.assert_allclose(recovered, points[:, :2], atol=3e-5)
        np.testing.assert_allclose(augmented[:, 2]/0.9, points[:, 2], atol=3e-5)
        self.assertEqual(transformed_masks.shape, masks.shape)
        self.assertTrue(set(np.unique(transformed_masks)).issubset({0, 1}))


if __name__ == '__main__':
    unittest.main()
