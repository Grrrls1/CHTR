"""Behavior and gradient contracts for the separate free-rollout implementation."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from models.networks.model import StrokeTrajectoryModel
from models.networks.autoregressive import generate
from models.networks.free_rollout import rollout, visual_predictions
from models.losses.free_rollout_loss import trajectory_loss,visual_loss,weighted_loss

CONFIG=dict(geometry_loss=dict(type='pixel_charbonnier',epsilon_px=.5,scale_px=10.),
            alignment_points=8,rollout=dict(max_strokes=3,max_points=6))


class FreeRolloutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(12)
        self.model=StrokeTrajectoryModel(dim=16,heads=2,stroke_layers=1,point_layers=1,
            dropout=0,memory_size=2,max_strokes=3,max_points=6,separate_branches=True).eval()
        self.image=torch.rand(1,1,32,32)
        self.sample=dict(image=self.image[0],image_size=32,
            points=torch.tensor([[.1,.2,.1,1.,0,0],[.2,.2,.1,0,1,0],
                                 [.5,.4,.2,1,0,0],[.6,.5,.2,0,0,1]]),
            stroke_offsets=torch.tensor([0,2,4]),stroke_types=torch.tensor([1,2]),
            stroke_masks=torch.ones(2,32,32))

    def force_event(self,event):
        with torch.no_grad():
            self.model.state_head.weight.zero_();self.model.state_head.bias.fill_(-8)
            self.model.state_head.bias[event]=8

    def test_matches_existing_greedy_and_all_limits(self):
        for event,reason in [(0,'max_points_reached'),(1,'max_strokes_reached'),(2,None)]:
            self.force_event(event)
            r=rollout(self.model,self.image)
            old=generate(self.model,self.image)
            geometry=torch.cat([s['geometry'] for s in r['strokes']]).detach().clone()
            geometry[:,:2]*=31;geometry[:,2]*=32
            torch.testing.assert_close(geometry,torch.from_numpy(old['points'][:,:3]))
            self.assertEqual(r['truncation_reason'],reason)
            self.assertEqual(r['completed'],old['completed'])
            self.assertFalse(r['used_ground_truth'])
            self.assertEqual([len(s['geometry']) for s in r['strokes']],
                             [b-a for a,b in zip(old['stroke_offsets'][:-1],old['stroke_offsets'][1:])])

    def test_target_change_only_changes_loss(self):
        self.force_event(2)
        a=rollout(self.model,self.image)
        loss_a=weighted_loss(trajectory_loss(a,self.sample,CONFIG),CONFIG)
        sample=dict(self.sample);sample['points']=self.sample['points'].clone();sample['points'][:,:2]+=.2
        b=rollout(self.model,self.image)
        torch.testing.assert_close(a['strokes'][0]['geometry'],b['strokes'][0]['geometry'])
        loss_b=weighted_loss(trajectory_loss(b,sample,CONFIG),CONFIG)
        self.assertNotEqual(float(loss_a),float(loss_b))

    def test_future_loss_backpropagates_to_earlier_prediction(self):
        self.force_event(0)
        from models.networks.autoregressive import point_step as original
        steps=[]
        def traced(*args,**kwargs):
            g,state=original(*args,**kwargs);g.retain_grad();steps.append(g);return g,state
        with patch('models.networks.free_rollout.point_step',side_effect=traced):
            r=rollout(self.model,self.image)
        r['strokes'][0]['geometry'][-1].sum().backward()
        self.assertIsNotNone(steps[0].grad)
        self.assertGreater(float(steps[0].grad.abs().sum()),0)
        for name,p in self.model.named_parameters():
            if name.startswith(('encoder.','stroke_decoder.','mask_query.','type_head.')):
                self.assertIsNone(p.grad,name)
        self.assertGreater(float(self.model.type_embedding.grad.abs().sum()),0)

    def test_missing_and_extra_strokes_have_state_gradients(self):
        for event,term,index in [(2,'missing',0),(1,'extra',1)]:
            self.model.zero_grad(set_to_none=True);self.force_event(event)
            r=rollout(self.model,self.image);logits=r['strokes'][index]['state_logits'];logits.retain_grad()
            terms=trajectory_loss(r,self.sample,CONFIG)
            self.assertGreater(float(terms[term]),0)
            terms[term].backward()
            self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_visual_supervision_is_separate(self):
        self.force_event(2)
        r=rollout(self.model,self.image)
        self.assertEqual(len(r['strokes']),1)
        visual=visual_predictions(self.model,self.image,2)
        weighted_loss(visual_loss(visual,self.sample),CONFIG).backward()
        self.assertIsNone(self.model.geometry_head.weight.grad)
        self.assertGreater(float(self.model.encoder.stem[0].weight.grad.abs().sum()),0)

    def test_combined_loss_backward_finite(self):
        self.force_event(0)
        r=rollout(self.model,self.image)
        terms=trajectory_loss(r,self.sample,CONFIG)
        terms.update(visual_loss(visual_predictions(self.model,self.image,2),self.sample))
        loss=weighted_loss(terms,CONFIG);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for p in self.model.parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())

    def test_rejects_invalid_limits_and_dropout(self):
        with self.assertRaises(ValueError):rollout(self.model,self.image,max_points=7)
        with self.assertRaises(ValueError):rollout(self.model,self.image.expand(2,-1,-1,-1))
        self.model.point_layers[0].drop.p=.1
        with self.assertRaises(ValueError):rollout(self.model,self.image)


if __name__=='__main__':unittest.main()
