import math
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent))
from models.losses.pixel_geometry import geometry_losses,geometry_metrics
from models.losses.multitask import multitask_loss,DEFAULT_WEIGHTS

CONFIG=dict(type='pixel_charbonnier',epsilon_px=.5,scale_px=10.)


class PixelGeometryTests(unittest.TestCase):
    def test_pixel_units_stroke_balancing_and_displacement(self):
        pred=torch.zeros(2,4,3,requires_grad=True);target=torch.zeros(2,4,6)
        with torch.no_grad():pred[0,0,0]=.1;pred[1,:,0]=.2
        valid=torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
        r=geometry_losses(pred,target,valid,(1,1,101,101),CONFIG)
        rho10=(math.sqrt(100+.25)-.5)/10;rho20=(math.sqrt(400+.25)-.5)/10
        self.assertAlmostEqual(float(r['xy']),((rho10/2)+rho20)/2,places=6)
        self.assertAlmostEqual(float(r['start']),(rho10+rho20)/2,places=6)
        self.assertAlmostEqual(float(r['delta']),rho10/2,places=6)
        total=r['xy']+2*r['start']+.25*r['delta'];total.backward()
        self.assertTrue(torch.isfinite(pred.grad).all());self.assertGreater(float(pred.grad[0,0,0]),0)
        self.assertEqual(float(pred.grad[0,2:].abs().sum()),0.)

    def test_width_pixel_scale_balancing_padding_and_gradient(self):
        for size in (101, 256):
            pred=torch.zeros(2,4,3,requires_grad=True)
            target=torch.zeros(2,4,6)
            valid=torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
            with torch.no_grad():
                pred[0,0,2]=10/size
                pred[0,2:,2]=1000
                pred[1,:,2]=20/size
                pred[0,0,0]=10/(size-1)
                pred[1,:,0]=20/(size-1)
            result=geometry_losses(pred,target,valid,(1,1,size,size),CONFIG)
            rho10=(math.sqrt(100+.25)-.5)/10
            rho20=(math.sqrt(400+.25)-.5)/10
            self.assertAlmostEqual(float(result['width']),(rho10/2+rho20)/2,places=6)
            torch.testing.assert_close(result['width'],result['xy'])
            result['width'].backward()
            self.assertTrue(torch.isfinite(pred.grad).all())
            self.assertGreater(float(pred.grad[0,0,2]),0)
            self.assertEqual(float(pred.grad[0,2:].abs().sum()),0.)
            self.assertEqual(float(pred.grad[...,:2].abs().sum()),0.)

    def test_padding_and_single_point_strokes(self):
        target=torch.zeros(1,4,6);pred=torch.zeros(1,4,3,requires_grad=True)
        valid=torch.tensor([[1,0,0,0]],dtype=torch.bool)
        with torch.no_grad():pred[0,0,0]=.1;pred[0,1:]=1000000
        r=geometry_losses(pred,target,valid,(1,1,101,101),CONFIG)
        self.assertAlmostEqual(float(r['xy']),float(r['start']),places=6)
        self.assertEqual(float(r['delta']),0.)
        (r['xy']+r['start']+r['delta']).backward()
        self.assertEqual(float(pred.grad[0,1:].abs().sum()),0.)

    def test_start_loss_is_not_diluted_by_stroke_length(self):
        starts=[]
        for length in (2,100):
            pred=torch.zeros(1,length,3);pred[0,0,0]=.1
            r=geometry_losses(pred,torch.zeros(1,length,6),torch.ones(1,length,dtype=torch.bool),(1,1,101,101),CONFIG)
            starts.append(float(r['start']))
        self.assertEqual(starts[0],starts[1])

    def test_exact_match_and_invalid_settings(self):
        pred=torch.zeros(1,2,3,requires_grad=True);target=torch.zeros(1,2,6);valid=torch.ones(1,2,dtype=torch.bool)
        r=geometry_losses(pred,target,valid,(1,1,256,256),CONFIG)
        total=sum(r.values());self.assertEqual(float(total),0.);total.backward()
        self.assertEqual(float(pred.grad.abs().sum()),0.)
        for config in (dict(CONFIG,epsilon_px=0),dict(CONFIG,scale_px=float('nan')),dict(type='unknown')):
            with self.assertRaises(ValueError):geometry_losses(pred,target,valid,(1,1,256,256),config)

    def test_multitask_legacy_and_new_weighted_formula(self):
        points=torch.tensor([[[[.1,.2,.1,1,0,0],[.2,.3,.1,1,0,0],[.3,.4,.1,0,0,1]]]])
        valid=torch.ones(1,1,3,dtype=torch.bool)
        batch=dict(images=torch.zeros(1,1,101,101),points=points,valid_point=valid,
                   types=torch.zeros(1,1,dtype=torch.long),last=torch.ones(1,1),
                   endpoints=torch.zeros(1,1,4),masks=torch.zeros(1,1,4,4))
        geom=torch.tensor([[[.2,.3,.2],[.25,.4,.2],[.35,.5,.2]]],requires_grad=True)
        out=dict(geometry=geom,owners=torch.tensor([0]),strokes=torch.tensor([0]),
                 state_logits=torch.zeros(1,3,3),type_logits=torch.zeros(1,1,24),
                 last_logits=torch.zeros(1,1),endpoints=torch.zeros(1,1,4),mask_logits=torch.zeros(1,1,4,4))
        legacy,old=multitask_loss(out,batch)
        torch.testing.assert_close(old['xy'],F.smooth_l1_loss(geom[...,:2],points[0,...,:2]))
        torch.testing.assert_close(old['width'],F.smooth_l1_loss(geom[...,2],points[0,...,2]))
        self.assertNotIn('start',old)
        weights=dict(xy=1.,start=2.,delta=.25)
        loss,terms=multitask_loss(out,batch,weights,geometry_config=CONFIG)
        expected=sum(dict(DEFAULT_WEIGHTS,**weights)[k]*v for k,v in terms.items())
        torch.testing.assert_close(loss,expected)
        direct=geometry_losses(geom,points[0],valid[0],batch['images'].shape,CONFIG)
        torch.testing.assert_close(terms['width'],direct['width'])
        self.assertGreater(float(terms['width']),float(old['width']))
        self.assertIn('start',terms);loss.backward();self.assertTrue(torch.isfinite(geom.grad).all())
        metrics=geometry_metrics(geom,points[0],valid[0],batch['images'].shape)
        self.assertAlmostEqual(float(metrics['start_l2_px']),math.sqrt(200),places=5)
        self.assertGreater(float(metrics['xy_l2_px']),1)


if __name__=='__main__':unittest.main()
