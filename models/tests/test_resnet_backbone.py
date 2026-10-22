"""Full ResNet stages, FPN contracts, and free-rollout training integration."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.networks.model import CNNFPN
from models.networks.autoregressive import encode_image
from models.training.free_rollout_setup import create_model
from models.scripts.train_free_rollout import train_microbatch

ROOT = Path(__file__).resolve().parents[1]


class ResNetBackboneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_full_four_stage_pyramid_and_deepest_stage_gradients(self):
        encoder = CNNFPN(128, backbone='resnet18', pretrained=False).eval()
        image = torch.rand(1,1,256,256)
        pyramid = encoder(image)
        self.assertEqual([tuple(t.shape) for t in pyramid],
                         [(1,128,64,64),(1,128,32,32),(1,128,16,16),(1,128,8,8)])
        self.assertEqual(encoder.stages[-1][-1].conv2.out_channels,512)
        pyramid[0].square().mean().backward()
        grad = encoder.stages[-1][-1].conv2.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(),0.)

    def test_default_random_resnet_rollout_backward_and_condition_shapes(self):
        config = json.loads((ROOT/'configs/free_rollout.json').read_text())
        config['rollout'] = dict(max_strokes=2,max_points=3)
        # Keep the actual dim128 architecture. Avoid downloading any weights.
        with patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('unexpected download')):
            model, checkpoint = create_model(config,'cpu')
        self.assertIsNone(checkpoint)
        model.train()
        images = torch.rand(2,1,64,64)
        pixels,memory,sizes = encode_image(model,images)
        self.assertEqual(tuple(pixels.shape),(2,128,16,16))
        self.assertEqual(tuple(memory.shape),(2,80,128))
        self.assertEqual(sizes,[8,4])
        del pixels,memory
        with torch.no_grad():
            model.state_head.weight.zero_()
            model.state_head.bias.copy_(torch.tensor([-8.,-8.,8.]))
        samples = [dict(image=image,image_size=64,
            points=torch.tensor([[.1,.2,.1,1.,0,0],[.3,.4,.15,0,0,1]]),
            stroke_offsets=torch.tensor([0,2]),stroke_types=torch.tensor([1]),
            stroke_masks=torch.ones(1,64,64)) for image in images]
        log = train_microbatch(model,samples,config,2)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in log.values()))
        self.assertEqual(log['completed'],1.)
        for module in (model.encoder.stages[-1],model.geometry_head,model.state_head):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads),0.)


if __name__ == '__main__':
    unittest.main()
