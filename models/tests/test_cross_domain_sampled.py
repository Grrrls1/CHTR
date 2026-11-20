"""CPU integration checks for the sampled cross-domain protocol."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from models.scripts import evaluate_cross_domain_sampled as e
from models.networks.two_stage import recover_trajectories


class ProtocolTests(unittest.TestCase):
    def test_shared_test_only_random_manifests(self):
        ids=['test_%03d'%i for i in range(200)]
        a=e.sample_manifests(ids,100); b=e.sample_manifests(list(reversed(ids)),100)
        self.assertEqual(a,b)
        for m in a:
            self.assertEqual(len(set(m['ids'])),100)
            self.assertTrue(set(m['ids'])<=set(ids))
            self.assertFalse(m['full_split'])
        with self.assertRaises(ValueError): e.sample_manifests(ids,201)
        # The runner reads the one domain manifest for all model IDs.
        self.assertEqual(len(set(s for m in a for s in m['ids'])),len(set(s for m in b for s in m['ids'])))

    def test_cache_identity_atomic_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sample.json'
            r=dict(identity='v1',sample_id='x',committed=True,prediction=dict(points_xyw_px=[],stroke_offsets=[0]))
            e.write_json(p,r)
            self.assertTrue(e.valid_cache(e.read_json(p),'v1','x'))
            self.assertFalse(e.valid_cache(r,'v2','x'))
            self.assertFalse(e.valid_cache(dict(r,committed=False),'v1','x'))
            self.assertFalse((p.parent/'sample.json.tmp').exists())
            self.assertIsNone(e.cached_record(p,'v1','x'))  # incomplete metrics payload
            p.write_text('{partial')
            self.assertIsNone(e.cached_record(p,'v1','x'))
            r.update(metrics={'empty_output':1})
            e.write_json(p,r)
            self.assertEqual(e.cached_record(p,'v1','x'),r)

    def fixture(self,empty=False,truncated=True):
        image=np.full((256,256),255,np.uint8);image[10:20,10:20]=0
        points=np.array([[10,10,8],[20,20,8]],np.float32)
        masks=np.zeros((1,256,256),bool);masks[0,10:20,10:20]=True
        paths=[] if empty else [dict(points=torch.tensor(points/[255,255,256],dtype=torch.float32),
                                    terminated=False,truncated=truncated,stop_reason='point_cap',next_validity=.9)]
        m=len(paths)
        inst=dict(masks=torch.tensor(masks[:m]),starts=torch.zeros(m,3),boxes=torch.zeros(m,4),
                  confidences=torch.ones(m),stop_reason='stroke_cap',terminated=False,truncated=True)
        result=dict(instances=inst,strokes=paths,terminated=False,truncated=True)
        return image,points,np.array([0,2]),masks,result

    def test_empty_and_truncation_preserved(self):
        row,pred,*_=e.score(*self.fixture(empty=True))
        self.assertEqual(row['aiou'],0); self.assertEqual(row['character_trajectory_iou'],0)
        self.assertIsNone(row['ldtw_px']);self.assertEqual(row['missing_strokes'],1)
        self.assertEqual(pred['stroke_offsets'],[0]);self.assertTrue(pred['truncated'])
        row,pred,*_=e.score(*self.fixture())
        self.assertEqual(row['truncated'],1);self.assertEqual(row['character_truncated'],1)
        self.assertEqual(len(pred['points_xyw_px']),2)

    def test_repeat_before_sample_sd_and_undefined(self):
        rows=[]
        for empty in [True,False]:
            row,*_=e.score(*self.fixture(empty=empty));row.update(gt_exceeds_stroke_cap=0,gt_strokes_exceeding_point_cap=0,inference_seconds=1,peak_allocated_bytes=1,peak_reserved_bytes=2)
            rows.append(row)
        a=e.aggregate(rows)
        self.assertEqual(a['dtw_undefined_images'],1)
        self.assertEqual(a['character_truncation_denominator'],2)
        self.assertEqual(a['point_truncation_denominator'],1)
        self.assertAlmostEqual(a['ldtw_px'],rows[1]['ldtw_px'])
        s=e.mean_std([dict(x=x) for x in [0,1,2,3,4]])
        self.assertEqual(s['x_mean'],2);self.assertAlmostEqual(s['x_sd'],np.sqrt(2.5))

    def test_cross_domain_image_only_and_native_caps(self):
        class Stroke:
            training=False
            def generate(self,images,max_strokes,threshold):
                self.args=(images.shape,max_strokes,threshold)
                return [dict(starts=torch.empty(0,3),masks=torch.empty(0,256,256),terminated=False,truncated=True)]
        class Point:
            training=False
            def generate(self,*args,**kwargs): raise AssertionError('empty instances need no point calls')
        s=Stroke()
        result=recover_trajectories(s,Point(),torch.ones(1,1,256,256),max_strokes=32,max_points=256,point_batch_size=16)
        self.assertEqual(s.args,(torch.Size([1,1,256,256]),32,.5))
        self.assertEqual(result[0]['strokes'],[]);self.assertTrue(result[0]['truncated'])

if __name__=='__main__': unittest.main()
