"""Check gradient isolation in both directions, without changing predictions."""
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.networks.model import StrokeTrajectoryModel
from models.losses.multitask import multitask_loss


class BranchSeparationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        self.model = StrokeTrajectoryModel(dim=16, heads=2, stroke_layers=1, point_layers=1,
            dropout=0, memory_size=2, separate_branches=True)
        points = torch.rand(1, 2, 4, 6)
        points[..., 3:] = F.one_hot(torch.zeros(1, 2, 4, dtype=torch.long), 3)
        self.batch = dict(images=torch.rand(1, 1, 32, 32), points=points,
            valid_stroke=torch.ones(1, 2, dtype=torch.bool),
            valid_point=torch.ones(1, 2, 4, dtype=torch.bool),
            masks=(torch.rand(1, 2, 32, 32) > .5).float(),
            types=torch.tensor([[0, 1]]), last=torch.tensor([[0., 1.]]),
            endpoints=torch.rand(1, 2, 4))
        self.visual = ('encoder.', 'stroke_bos', 'stroke_position.', 'stroke_decoder.',
                        'mask_query.', 'type_head.', 'last_head.', 'endpoint_head.')

    def test_point_losses_do_not_update_any_visual_parameters(self):
        _, terms = multitask_loss(self.model(self.batch), self.batch)
        sum(terms[k] for k in ('xy', 'width', 'state', 'delta')).backward()
        for name, p in self.model.named_parameters():
            if name.startswith(self.visual):
                self.assertIsNone(p.grad, name)
        for name in ('type_embedding', 'condition.weight', 'point_layers.0.cross_attn.in_proj_weight',
                     'geometry_head.weight', 'state_head.weight'):
            p = dict(self.model.named_parameters())[name]
            self.assertGreater(float(p.grad.abs().sum()), 0, name)

    def test_visual_losses_do_not_update_point_parameters(self):
        _, terms = multitask_loss(self.model(self.batch), self.batch)
        sum(terms[k] for k in ('mask_bce', 'mask_dice', 'type', 'last', 'endpoints')).backward()
        for name, p in self.model.named_parameters():
            if not name.startswith(self.visual):
                self.assertIsNone(p.grad, name)
        for module in (self.model.encoder, self.model.stroke_decoder, self.model.mask_query, self.model.type_head):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)

    def test_separation_does_not_change_forward_values(self):
        self.model.eval()
        separated = self.model(self.batch)
        self.model.separate_branches = False
        coupled = self.model(self.batch)
        for key in separated:
            torch.testing.assert_close(separated[key], coupled[key])


if __name__ == '__main__':
    unittest.main()
