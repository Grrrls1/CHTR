"""Contracts for the separate three-task model and its original-point losses."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from models.networks.hierarchical_ar import HierarchicalAR,export_trajectory
from models.losses.hierarchical_ar_loss import three_task_loss,balanced_end_loss,geometry_metrics

MODEL=dict(dim=16,heads=2,stroke_layers=1,point_layers=1,backbone='cnn',memory_size=2,max_strokes=3,max_points=6)


def samples_for(lengths,images):
    rows=[]
    for counts,image in zip(lengths,images):
        points=torch.rand(sum(counts),3);points[:,2]=points[:,2]*.1+.02
        offsets=torch.tensor([0]+torch.tensor(counts).cumsum(0).tolist())
        rows.append(dict(image=image,points=points,stroke_offsets=offsets,image_size=image.shape[-1]))
    return rows


class HierarchicalARTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(20)
        self.model=HierarchicalAR(**MODEL).eval()
        self.images=torch.rand(3,1,32,32)
        self.lengths=[[1,3],[4],[2,1,5]]
        self.samples=samples_for(self.lengths,self.images)

    def force(self,point,stroke):
        with torch.no_grad():
            for head,value in ((self.model.point_end_head,point),(self.model.stroke_end_head,stroke)):
                head.weight.zero_();head.bias.fill_(value)

    def test_only_three_output_heads_and_no_auxiliary_parameters(self):
        names=[name for name,_ in self.model.named_parameters()]
        self.assertFalse(any(any(word in name for word in ('mask','type','endpoint','state_head')) for name in names))
        self.assertEqual(self.model.point_embedding.in_features,3)
        self.assertEqual(self.model.point_end_head.out_features,1)
        self.assertEqual(self.model.stroke_end_head.out_features,1)

    def test_training_ignores_early_stop_and_keeps_every_original_point(self):
        self.force(20,20)
        outputs=self.model(self.images,self.lengths)
        self.assertEqual([[len(s['geometry']) for s in row['strokes']] for row in outputs],self.lengths)
        total,terms=three_task_loss(outputs,self.samples)
        total.backward()
        self.assertEqual(set(terms),{'geo','point_end','stroke_end'})
        for head in (self.model.geometry_head,self.model.point_end_head,self.model.stroke_end_head):
            self.assertGreater(head.bias.grad.abs().sum().item(),0.)
        for name,param in self.model.encoder.named_parameters():
            self.assertIsNotNone(param.grad,name)

    def test_geometry_reaches_backbone_stroke_and_future_history(self):
        original=self.model.point_step;steps=[]
        def traced(*args,**kwargs):
            geometry,end=original(*args,**kwargs);geometry.retain_grad();steps.append(geometry);return geometry,end
        with patch.object(self.model,'point_step',side_effect=traced):
            out=self.model(self.images[:1],[[4]])
        out[0]['strokes'][0]['geometry'][-1].sum().backward()
        self.assertGreater(steps[0].grad.abs().sum().item(),0.)
        for module in (self.model.encoder,self.model.stroke_decoder,self.model.condition):
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads);self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads),0.)

    def test_ragged_batch_paths_loss_and_gradients_match_serial(self):
        other=copy.deepcopy(self.model)
        batch=self.model(self.images,self.lengths)
        serial=[other(image[None],[counts])[0] for image,counts in zip(self.images,self.lengths)]
        for a,b in zip(batch,serial):
            torch.testing.assert_close(a['stroke_end_logits'],b['stroke_end_logits'],rtol=1e-4,atol=1e-5)
            for sa,sb in zip(a['strokes'],b['strokes']):
                torch.testing.assert_close(sa['geometry'],sb['geometry'],rtol=1e-4,atol=1e-5)
        la,_=three_task_loss(batch,self.samples);la.backward()
        for prediction,sample in zip(serial,self.samples):
            loss,_=three_task_loss([prediction],[sample]);(loss/len(serial)).backward()
        for (name,a),(_,b) in zip(self.model.named_parameters(),other.named_parameters()):
            self.assertEqual(a.grad is None,b.grad is None,name)
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=3e-3,atol=2e-4,msg=name)

    def test_singleton_and_balanced_binary_loss(self):
        logits=torch.tensor([-.2,.3,1.2])
        expected=(F.binary_cross_entropy_with_logits(logits[:-1],torch.zeros(2))+
                  F.binary_cross_entropy_with_logits(logits[-1:],torch.ones(1)))/2
        torch.testing.assert_close(balanced_end_loss(logits),expected)
        torch.testing.assert_close(balanced_end_loss(logits[-1:]),F.softplus(-logits[-1]))
        out=self.model(self.images[:1],[[1]])
        self.assertTrue(torch.isfinite(three_task_loss(out,samples_for([[1]],self.images[:1]))[0]))

    def test_inference_stops_after_finishing_current_stroke_and_exports_events(self):
        self.force(-20,20)
        result=self.model.generate(self.images[:1])
        self.assertEqual(result['truncation_reason'],'max_points_reached')
        self.assertFalse(result['completed'])
        self.assertEqual(len(result['strokes'][0]['geometry']),6)
        raw=export_trajectory(result,32)
        self.assertEqual(raw['points'][-1,3:].tolist(),[1,0,0])
        self.force(20,20)
        result=self.model.generate(self.images[:1])
        self.assertTrue(result['completed']);self.assertEqual(len(result['strokes'][0]['geometry']),1)
        self.assertEqual(export_trajectory(result,32)['points'][-1,3:].tolist(),[0,0,1])
        self.force(20,-20)
        result=self.model.generate(self.images[:1])
        self.assertEqual(result['truncation_reason'],'max_strokes_reached')
        self.assertEqual(len(result['strokes']),3)
        self.assertEqual(export_trajectory(result,32)['points'][-1,3:].tolist(),[0,1,0])

    def test_mixed_point_and_stroke_stopping_order(self):
        original=self.model.point_step;calls=[]
        def end_after_third(*args,**kwargs):
            g,end=original(*args,**kwargs);calls.append(args[1]);return g,end*0+(-20 if args[1]<2 else 20)
        self.force(-20,20)
        with patch.object(self.model,'point_step',side_effect=end_after_third):result=self.model.generate(self.images[:1])
        self.assertEqual(calls,[0,1,2]);self.assertTrue(result['completed'])

    def test_geometry_targets_never_enter_forward_or_change_predictions(self):
        a=self.model(self.images,self.lengths)
        changed=copy.deepcopy(self.samples);changed[0]['points'][1:,:2]+=.1
        b=self.model(self.images,self.lengths)
        for sa,sb in zip(a[0]['strokes'],b[0]['strokes']):torch.testing.assert_close(sa['geometry'],sb['geometry'])
        self.assertNotEqual(three_task_loss(a,self.samples)[1]['geo'].item(),three_task_loss(b,changed)[1]['geo'].item())

    def test_training_rejects_caps_instead_of_truncating(self):
        for counts in ([[7]],[[1,1,1,1]],[[0]]):
            with self.assertRaises(ValueError):self.model(self.images[:1],counts)
        with self.assertRaises(ValueError):self.model.generate(self.images[:1],point_end_threshold=0.)

    def test_batch24_uses_parallel_axis(self):
        images=self.images[:1].expand(24,-1,-1,-1)
        outputs=self.model(images,[[2]]*24)
        self.assertEqual(len(outputs),24)
        self.assertTrue(all(row['strokes'][0]['geometry'].shape==(2,3) for row in outputs))

    def test_pixel_metrics_are_separate_from_normalized_loss(self):
        out=self.model(self.images[:1],[[2]])
        sample=samples_for([[2]],self.images[:1])[0]
        sample['points']=out[0]['strokes'][0]['geometry'].detach().clone()
        metrics=geometry_metrics(out,[sample])
        self.assertTrue(all(value==0 for value in metrics.values()))


if __name__=='__main__':unittest.main()
