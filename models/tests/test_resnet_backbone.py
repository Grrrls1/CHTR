"""Legacy baseline ResNet stages and FPN gradient contracts."""
from pathlib import Path
import sys
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.networks.model import CNNFPN


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


if __name__ == '__main__':
    unittest.main()
