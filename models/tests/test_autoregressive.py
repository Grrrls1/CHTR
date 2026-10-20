from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent))
from models.networks.model import StrokeTrajectoryModel
from models.networks.autoregressive import encode_image,next_stroke,point_step,generate


class AutoregressiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        self.model=StrokeTrajectoryModel(dim=16,heads=2,stroke_layers=1,point_layers=2,
                                         dropout=0,memory_size=2,max_strokes=3,max_points=8).eval()
        self.image=torch.rand(1,1,32,32)

    def test_cached_logits_equal_teacher_forced_forward(self):
        model=self.model
        points=torch.rand(1,2,5,6);points[...,3:]=F.one_hot(torch.zeros(1,2,5,dtype=torch.long),3)
        batch=dict(images=self.image,points=points,valid_stroke=torch.ones(1,2,dtype=torch.bool),
                   valid_point=torch.ones(1,2,5,dtype=torch.bool))
        with torch.no_grad():
            expected=model(batch);pixels,memory,sizes=encode_image(model,self.image);history=[]
            for stroke in range(2):
                z,mask,types,cond,bias=next_stroke(model,pixels,memory,sizes,history,32);history.append(z)
                torch.testing.assert_close(mask,expected['mask_logits'][:,stroke],atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(types,expected['type_logits'][:,stroke],atol=2e-6,rtol=2e-5)
                caches=[{} for _ in model.point_layers]
                for i in range(5):
                    geometry,state=point_step(model,None if i==0 else points[:,stroke,i-1],i,cond,memory,bias,caches)
                    torch.testing.assert_close(geometry[0],expected['geometry'][stroke,i],atol=2e-6,rtol=2e-5)
                    torch.testing.assert_close(state[0],expected['state_logits'][stroke,i],atol=2e-6,rtol=2e-5)

    def event(self,index):
        with torch.no_grad():
            self.model.state_head.weight.zero_();self.model.state_head.bias.fill_(-10)
            self.model.state_head.bias[index]=10

    def test_drawing_end_uses_only_image(self):
        self.event(2);result=generate(self.model,self.image)
        self.assertTrue(result['completed']);self.assertFalse(result['used_ground_truth'])
        self.assertEqual(result['stroke_offsets'],[0,1]);self.assertEqual(result['points'].shape,(1,6))
        np.testing.assert_array_equal(result['points'][-1,3:],[0,0,1])

    def test_point_limit_does_not_fabricate_end(self):
        self.event(0);result=generate(self.model,self.image,max_points=3)
        self.assertTrue(result['truncated']);self.assertEqual(result['truncation_reason'],'max_points_reached')
        np.testing.assert_array_equal(result['points'][-1,3:],[1,0,0])

    def test_stroke_limit_is_explicit(self):
        self.event(1);result=generate(self.model,self.image,max_strokes=2)
        self.assertEqual(result['stroke_offsets'],[0,1,2])
        self.assertEqual(result['truncation_reason'],'max_strokes_reached')
        np.testing.assert_array_equal(result['points'][-1,3:],[0,1,0])


if __name__=='__main__':unittest.main()
