"""Required synthetic protocol/integration checks; no GPU or training."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from models.experiments import mask_start_core as c
from models.experiments.mask_start_summary import collapse,summarize

class ProtocolTests(unittest.TestCase):
    def test_condition_count(self):
        a=c.conditions();self.assertEqual(len(a),54);self.assertEqual(sum(x['draws'] for x in a),198)
        self.assertEqual(sum(x['draws']==1 for x in a),18)
        self.assertEqual(len({x['condition_id'] for x in a}),54)

    def test_gaussian_variance_units_and_seed(self):
        z=np.array([c.standard_noise(42,'x',i,0) for i in range(6000)])
        self.assertTrue(np.all(np.abs(z.mean(0))<.05));self.assertTrue(np.all(np.abs(z.var(0)-1)<.06))
        self.assertTrue(np.array_equal(c.standard_noise(42,'x',3,4),c.standard_noise(42,'x',3,4)))
        self.assertFalse(np.array_equal(c.standard_noise(42,'x',3,4),c.standard_noise(42,'x',3,3)))
        residual=np.array([[1,2,3],[-1,-2,-3]],float);cal=c.calibrate(residual)
        self.assertAlmostEqual(cal['sigma_xy_base_px'],np.sqrt(2.5));self.assertEqual(cal['mu_w_base_px2'],9)

    def test_component_pairing_and_scale(self):
        cal=dict(sigma_xy_base_px=2,sigma_w_base_px=3);gt=[100,120,10];pred=[101,119,8]
        p10,n10=c.first_point(gt,pred,'N10',1,0,cal,42,'s',0)
        p01,n01=c.first_point(gt,pred,'N01',1,0,cal,42,'s',0)
        p11,n11=c.first_point(gt,pred,'N11',1,0,cal,42,'s',0)
        self.assertTrue(np.array_equal(p10[:2],p11[:2]));self.assertEqual(p01[2],p11[2])
        self.assertEqual(p10[2],10);self.assertTrue(np.array_equal(p01[:2],gt[:2]))
        _,twice=c.first_point(gt,pred,'N11',2,0,cal,42,'s',0)
        np.testing.assert_allclose(twice['requested_noise_px'],2*np.array(n11['requested_noise_px']))
        start,_=c.first_point(gt,pred,'N00',0,0,cal,42,'s',0);np.testing.assert_equal(start,gt)
        actual,_=c.first_point(gt,pred,'PRED',0,0,cal,42,'s',0);np.testing.assert_equal(actual,pred)

    def test_boundary_and_w_floor(self):
        cal=dict(sigma_xy_base_px=10000,sigma_w_base_px=10000)
        requests=[c.first_point([0,255,.001],[1,2,3],'N11',2,d,cal,42,'s',0) for d in range(30)]
        self.assertTrue(any(r[1]['xy_clipped'] for r in requests));self.assertTrue(any(r[1]['w_floored'] for r in requests))
        for a,meta in requests:
            self.assertTrue(np.all(a[:2]>=0) and np.all(a[:2]<=255));self.assertGreaterEqual(a[2],.001-1e-10)
            np.testing.assert_allclose((a/c.SCALE)*c.SCALE,a,atol=2e-5)

    def test_matching_is_geometric_and_deterministic(self):
        a=np.zeros((2,256,256),bool);a[0,20:30,20:30]=1;a[1,50:60,50:60]=1
        p=a[::-1].copy();pairs,_=c.matching(p,a)
        self.assertEqual([(r['pred_index'],r['gt_index']) for r in pairs],[(0,1),(1,0)])
        self.assertEqual(pairs,c.matching(p,a)[0]);self.assertEqual(c.matching(np.zeros((0,256,256),bool),a)[0],[])
        same=np.repeat(a[:1],2,axis=0);self.assertEqual(c.matching(same,same)[0],c.matching(same,same)[0])

    def test_first_point_mask_input_contract(self):
        import torch
        from models.networks.two_stage import decode_stroke_instances
        class Point:
            training=False
            def generate(self,images,starts,max_points,threshold):
                self.images=images.clone()
                return [dict(points=s[None].clone(),terminated=True,truncated=False,stop_reason='invalid') for s in starts]
        model=Point();image=torch.full((1,1,256,256),.25);mask=torch.zeros(1,256,256);mask[:,10:20,10:20]=1
        start=torch.tensor([[.3,.4,.02]])
        inst=dict(masks=mask,starts=start,terminated=True,truncated=False)
        a=decode_stroke_instances(model,image,[inst],400,.5,16)[0]
        b=decode_stroke_instances(model,image,[inst],400,.5,16)[0]
        self.assertTrue(torch.equal(a['strokes'][0]['points'][0],start[0]))
        self.assertTrue(torch.equal(a['strokes'][0]['points'],b['strokes'][0]['points']))
        self.assertEqual(float(model.images[0,0,0,0]),1);self.assertEqual(float(model.images[0,0,15,15]),.25)

    def test_arc_recovery_and_degenerate(self):
        path=np.array([[10,10,8],[20,10,9],[30,10,10]],np.float32)
        noisy=path.copy();noisy[0]+=[3,4,2]
        r=c.response(path,path,noisy,[3,4,2])
        self.assertEqual(r['xy_clean_sq'][0],25);self.assertEqual(r['w_clean_sq'][0],4)
        self.assertEqual(r['xy_clean_sq'][-1],0)
        a,length,deg=c.arc64(path[:1]);self.assertTrue(deg);self.assertEqual(length,0);self.assertTrue(np.all(a==path[0]))
        repeated=np.repeat(path[:1],3,axis=0);self.assertTrue(c.arc64(repeated)[2])
        reverse=path[::-1];self.assertTrue(np.array_equal(c.arc64(reverse)[0][0],reverse[0]))

    def test_empty_and_count_truncation_preserved(self):
        gt=[np.array([[10,10,8],[20,10,8]],np.float32)];gm=np.array([c.raster(gt[0])])
        inst=dict(masks=np.zeros((0,256,256),bool),terminated=True,truncated=False)
        r=c.score_character(np.full((256,256),255,np.uint8),gt,gm,[],[],inst)
        self.assertEqual(r['image_char_iou'],0);self.assertIsNone(r['ldtw_px']);self.assertEqual(r['missing_strokes'],1)
        pred=gt.copy();st=[dict(terminated=False,truncated=True)];inst=dict(masks=gm,terminated=False,truncated=True)
        r=c.score_character(np.full((256,256),255,np.uint8),gt,gm,pred,st,inst)
        self.assertEqual(r['truncated'],1);self.assertEqual(r['character_truncated'],1)
        self.assertEqual(c.pack_prediction([],[])['stroke_offsets'],[0])

    def test_atomic_cache_and_mask_bits(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';c.write_json(p,dict(committed=True,x=1));self.assertEqual(c.read_json(p)['x'],1)
            self.assertFalse((Path(d)/'x.json.tmp').exists())
        a=np.random.default_rng(4).random((3,256,256))>.5
        np.testing.assert_array_equal(np.unpackbits(np.packbits(a,axis=-1),axis=-1,count=256).astype(bool),a)

    def test_draws_collapse_before_character_aggregation(self):
        path=np.array([[10,10,8],[20,10,8]],np.float32)
        r=c.instance_metrics(path,path,path,[0,0,0],dict(terminated=True,truncated=False));r.update(pred_index=0,gt_index=0)
        base=dict(sample_id='x',prediction=c.pack_prediction([path],[dict(terminated=True,truncated=False)]),
                  whole_character={'dtw_status':'ok','x':1.},instances=[r])
        import copy
        b=copy.deepcopy(base);b['whole_character']['x']=3.;b['instances'][0]['post_start']['aligned_xy_mae_px']=2.
        combined=collapse([[base],[b]])[0]
        self.assertEqual(combined['whole_character']['x'],2);self.assertEqual(combined['whole_character']['dtw_status'],'ok')
        self.assertEqual(combined['single_character']['post_start_aligned_xy_mae_px'],1.)


class AggregationIntegrationTests(unittest.TestCase):
    def test_full_base_matrix_and_remaining_manifest(self):
        import copy,csv
        from unittest.mock import patch
        from models.experiments.mask_start_summary import aggregate
        path=np.array([[10,10,8],[20,10,8]],np.float32);gt=[path]
        masks=np.array([c.raster(path)]);image=np.where(masks[0],0,255).astype(np.uint8)
        state=dict(terminated=True,truncated=False,stop_reason='invalid',first_point_exact=True)
        whole=c.score_character(image,gt,masks,[path],[state],dict(masks=masks,terminated=True,truncated=False))
        whole.update(inference_seconds=.5,peak_allocated_bytes=1,peak_reserved_bytes=2)
        inst=c.instance_metrics(path,path,path,[0,0,0],state);inst.update(pred_index=0,gt_index=0,input_first_xyw_px=path[0].tolist())
        class Fake:
            def __init__(self,out,gpu=False):
                self.out=out;self.cfg=dict(conditions=c.conditions(),repeats=list(range(20261120,20261125)))
            def ids(self,split,domain):return ['x','y']
            def assets(self,domain,sid):return image,gt,masks
            def e2_path(self,split,domain,stage,point,sid):return (split,f'E2_{domain}_{stage}_{point}',0,sid)
            def e1_path(self,split,cond,draw,sid):return (split,cond['condition_id'],draw,sid)
            def cached(self,p):
                split,cid,draw,sid=p
                r=dict(sample_id=sid,first_point_exact=True,prediction=c.pack_prediction([path],[state]))
                if cid.startswith('E2'):r['metrics']=copy.deepcopy(whole)
                else:
                    r.update(whole_character=copy.deepcopy(whole),instances=[copy.deepcopy(inst)],
                        single_character=c.single_character_metrics([inst]),coverage=dict(gt_strokes=1,pred_strokes=1,matched=1),conditional_point_gpu_seconds=.3)
                return r
            def close(self):pass
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);cfg=dict(conditions=c.conditions())
            c.write_json(out/'protocol.json',cfg)
            plan=[]
            for split in ['val','test']:
                for cond in c.conditions():
                    for draw in range(cond['draws']):plan.append(dict(split=split,condition_id=cond['condition_id'],draw=draw,state='planned'))
            c.write_csv(out/'execution_manifest.csv',plan)
            for domain in ['BOOK','CASIA']:
                for seed in range(20261120,20261125):c.write_json(out/'manifests'/domain/f'seed_{seed}.json',dict(ids=['x','y']))
            with patch('models.scripts.evaluate_mask_start_stage.Experiment',Fake), patch('models.experiments.mask_start_summary.baseline_reproduction'):
                aggregate(out,'base',out/'report.txt')
            import io
            read=lambda p:list(csv.DictReader(io.StringIO(p.read_text())))
            a=read(out/'execution_manifest.csv');self.assertEqual(len(a),396)
            self.assertEqual(sum(r['state']=='complete' for r in a),36)
            self.assertEqual(len(read(out/'summaries/base/per_repeat.csv')),108)
            self.assertEqual(len(read(out/'summaries/base/mean_std.csv')),36)
            self.assertTrue((out/'report.txt').exists())
            with patch('models.scripts.evaluate_mask_start_stage.Experiment',Fake), patch('models.experiments.mask_start_summary.noise_acceptance'), patch('models.experiments.mask_start_summary.plots_and_examples'), patch('models.experiments.mask_start_summary.mask_comparison'):
                aggregate(out,'final',out/'final_report.txt')
            final=read(out/'execution_manifest.csv');self.assertEqual(len(final),396)
            self.assertTrue(all(r['state']=='complete' for r in final))
            self.assertEqual(len(read(out/'summaries/final/per_execution_unit.csv')),396)
            self.assertEqual(len(read(out/'summaries/final/mean_std.csv')),108)
            self.assertEqual(len(read(out/'summaries/final/per_repeat.csv')),324)

if __name__=='__main__':unittest.main()
