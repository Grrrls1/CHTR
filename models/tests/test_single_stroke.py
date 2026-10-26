import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch

from models.datasets.dataset_split import create_split
from models.datasets.single_stroke import StrokeCatalog,SingleStrokeDataset,collate_single_strokes,masked_stroke_image
from models.losses.single_stroke import single_stroke_loss
from models.networks.single_stroke import SingleStrokeModel


def fixture(root):
    rows=[];assets={}
    image=np.tile(np.arange(256,dtype=np.uint8),(256,1))
    for c in ('永','一'):
        for i in range(3):
            sid=f'{c}{i}';p=np.array([[30,30,4],[40,40,6],[50,50,8],[60,60,10]],np.float32)
            off=np.array([0,1,4],np.int64)
            row=dict(sample_id=sid,character=c,stroke_count=2,image_key=sid+'/im',trajectory_key=sid+'/tr',masks_key=sid+'/mask',
                     stroke_map=[dict(stroke_index=k,point_range=off[k:k+2].tolist()) for k in range(2)])
            rows.append(row);assets[row['image_key']]=cv2.imencode('.png',image)[1].tobytes()
            buf=io.BytesIO();np.savez(buf,points=p,stroke_offsets=off,image_size=256);assets[row['trajectory_key']]=buf.getvalue()
            mask=np.full((256,256),255,np.uint8);mask[20:80,20:80]=0
            png=cv2.imencode('.png',mask)[1]
            buf=io.BytesIO();np.savez(buf,**{'stroke000.png':png,'stroke001.png':png});assets[row['masks_key']]=buf.getvalue()
    db=root/'db';db.mkdir()
    with lmdb.open(str(db),map_size=10*1024*1024) as e:
        with e.begin(write=True) as t:
            t.put(b'__ids__',json.dumps([r['sample_id'] for r in rows]).encode());t.put(b'__ids_ready__',json.dumps([r['sample_id'] for r in rows]).encode())
            for r in rows:t.put(r['sample_id'].encode(),json.dumps(r).encode())
            for k,v in assets.items():t.put(k.encode(),v)
    split=create_split(rows,[[r['sample_id']] for r in rows]);path=root/'split.json';path.write_text(json.dumps(split))
    return db,path


class DataTests(unittest.TestCase):
    def test_split_mask_alignment_augmentation_and_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            db,path=fixture(Path(tmp));catalog=StrokeCatalog(db,path)
            try:
                sets=[]
                for name in ('train','val','test'):
                    ds=SingleStrokeDataset(catalog,name,max_points=3)
                    sets.append({ds.key(i)[0] for i in range(len(ds))})
                    s=ds[1];self.assertEqual(s['points'].shape,(3,3))
                    self.assertEqual(float(s['image'][0,100,100]),1.)
                    self.assertAlmostEqual(float(s['image'][0,40,40]),40/255.)
                    batch=collate_single_strokes([ds[0],s])
                    self.assertEqual(batch['valid'].tolist(),[[False,False,False],[True,True,False]])
                    self.assertTrue(torch.equal(batch['targets'][1,0],s['points'][1]))
                    self.assertTrue(torch.equal(batch['history'][1,0],s['points'][0]))
                self.assertFalse(sets[0]&sets[1] or sets[1]&sets[2] or sets[0]&sets[2])
                with self.assertRaises(ValueError):SingleStrokeDataset(catalog,'val',augment=True)
                with self.assertRaises(ValueError):SingleStrokeDataset(catalog,'train',max_points=2)
                aug=SingleStrokeDataset(catalog,'train',max_points=3,augment=True)[1]
                plain=SingleStrokeDataset(catalog,'train',max_points=3)[1]
                raw=plain['points'].numpy()*[255,255,256]
                matrix=aug['transform'].numpy()
                expected=raw[:,:2]@matrix[:2,:2].T+matrix[:2,2]
                np.testing.assert_allclose(aug['points'].numpy()[:,:2]*255,expected,atol=1e-4)
                np.testing.assert_allclose(aug['points'].numpy()[:,2]*256,raw[:,2]*matrix[0,0],atol=1e-4)
            finally:catalog.close()


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(3)
        self.model=SingleStrokeModel(dim=16,heads=4,encoder_layers=1,max_points=8,dropout=0.)
        self.images=torch.rand(2,1,256,256)
        self.history=torch.rand(2,4,3);self.history[:,:,2]*=.1
        self.active=torch.tensor([[True]*4,[True,True,False,False]])

    def test_shapes_gradients_and_four_scales(self):
        self.model.train()
        memory=self.model.visual(self.images)
        self.assertEqual([m.shape[1] for m in memory],[64,256,1024,4096])
        out=self.model.decode(memory,self.history,self.active)
        self.assertEqual(out['points'].shape,(2,4,3))
        batch=dict(targets=torch.rand(2,4,3),valid=torch.tensor([[True,True,True,False],[True,False,False,False]]),active=self.active)
        loss,_=single_stroke_loss(out,batch);loss.backward()
        for name in ('width_head.weight','xy_head.weight','valid_head.weight','visual.stem.0.weight','point_embedding.0.weight'):
            grad=dict(self.model.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0.)

    def test_causality_padding_and_width_condition(self):
        self.model.eval()
        with torch.no_grad():
            memory=self.model.visual(self.images)
            a=self.model.decode(memory,self.history,self.active)
            changed=self.history.clone();changed[0,2:]=100;changed[1,2:]=-100
            b=self.model.decode(memory,changed,self.active)
            torch.testing.assert_close(a['points'][0,:2],b['points'][0,:2])
            torch.testing.assert_close(a['points'][1,:2],b['points'][1,:2])
            changed=self.history.clone();changed[0,0,2]+=1
            c=self.model.decode(memory,changed,self.active)
            self.assertGreater(float((a['points'][0,0]-c['points'][0,0]).abs().sum()),1e-6)
            batch=dict(targets=torch.zeros(2,4,3),valid=torch.zeros(2,4,dtype=torch.bool),active=self.active)
            _,terms=single_stroke_loss(a,batch)
            self.assertEqual(float(terms['xy']),0.);self.assertEqual(float(terms['width']),0.)

    def test_stop_cap_and_checkpoint(self):
        self.model.eval();first=self.history[:,0]
        with torch.no_grad():self.model.valid_head.weight.zero_();self.model.valid_head.bias.fill_(-20)
        result=self.model.generate(self.images,first,max_points=3)
        for i,r in enumerate(result):
            self.assertTrue(r['terminated']);self.assertFalse(r['truncated']);self.assertEqual(len(r['points']),1)
            self.assertTrue(torch.equal(r['points'][0],first[i]))
        with torch.no_grad():self.model.valid_head.bias.fill_(20)
        result=self.model.generate(self.images,first,max_points=3)
        self.assertTrue(all(r['truncated'] and len(r['points'])==3 for r in result))
        other=SingleStrokeModel(dim=16,heads=4,encoder_layers=1,max_points=8,dropout=0.).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';torch.save(self.model.state_dict(),path);other.load_state_dict(torch.load(path))
        with torch.no_grad():
            a=self.model(self.images,self.history,self.active);b=other(self.images,self.history,self.active)
        torch.testing.assert_close(a['points'],b['points'],rtol=0,atol=0)

    def test_batched_generation_matches_individual(self):
        self.model.eval()
        first=self.history[:,0]
        with torch.no_grad():self.model.valid_head.weight.zero_();self.model.valid_head.bias.fill_(20)
        batch=self.model.generate(self.images,first,max_points=4)
        for i,result in enumerate(batch):
            single=self.model.generate(self.images[i:i+1],first[i:i+1],max_points=4)[0]
            torch.testing.assert_close(result['points'],single['points'],rtol=1e-5,atol=1e-6)
            self.assertEqual(result['stop_reason'],single['stop_reason'])
        original_decode=self.model.decode
        def mixed_decode(memory,history,active):
            out=original_decode(memory,history,active)
            lengths=torch.where(history[:,0,0]<.5,2,3)
            out['valid_logits'][:,-1]=torch.where(lengths<=history.shape[1],-20.,20.)
            return out
        self.model.decode=mixed_decode
        first[0,0]=.2;first[1,0]=.8
        mixed=self.model.generate(self.images,first,max_points=4)
        self.assertEqual([len(r['points']) for r in mixed],[2,3])
        self.assertTrue(all(r['terminated'] for r in mixed))
        for i,result in enumerate(mixed):torch.testing.assert_close(result['points'][0],first[i])


class EvaluationTests(unittest.TestCase):
    def test_full_split_free_metrics_and_fixed_character_gallery(self):
        from models.training.single_stroke_evaluation import evaluate_split,visualization_ids
        class FirstOnly(torch.nn.Module):
            max_points=4
            def forward(self,images,history,active):
                return dict(points=torch.zeros_like(history),valid_logits=torch.zeros_like(active,dtype=torch.float32))
            def generate(self,images,first_points,max_points,threshold):
                # This signature deliberately accepts no GT points or lengths.
                return [dict(points=p[None].clone(),terminated=True,truncated=False,stop_reason='invalid') for p in first_points]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);db,split=fixture(root);catalog=StrokeCatalog(db,split)
            try:
                val=SingleStrokeDataset(catalog,'val',max_points=4)
                fixed=visualization_ids(val,30)
                self.assertEqual(len(fixed),2)
                self.assertEqual(fixed,visualization_ids(val,30))
                c=dict(eval_batch_size=3,decode_max_points=4,validity_threshold=.5,width_weight=1.,valid_weight=.2,
                       model=dict(max_points=4))
                model=FirstOnly().train()
                for name in ('first','second'):
                    report=evaluate_split(model,val,'cpu',c,root/name,fixed)
                    self.assertTrue(model.training)
                    self.assertTrue(report['full_split']);self.assertEqual(report['strokes'],4)
                    self.assertEqual(report['visualization_ids'],fixed)
                    self.assertEqual(report['free']['terminated_rate'],1.)
                    self.assertEqual(report['free']['point_count_mae'],1.)
                    self.assertEqual(report['visualized_characters'],2)
                    self.assertEqual(len(list((root/name).rglob('*.npz'))),4)
                    self.assertEqual(len((root/name/'strokes.jsonl').read_text().splitlines()),4)
                    self.assertTrue(all((root/name/sid/'index.html').exists() for sid in fixed))
            finally:catalog.close()


class TrainingIntegrationTests(unittest.TestCase):
    def test_epoch_only_validation_and_final_full_test(self):
        import subprocess
        import sys
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);db,split=fixture(tmp)
            config=json.loads((root/'models/configs/single_stroke_smoke.json').read_text())
            config.update(dataset=str(db),split_file=str(split),epochs=2,train_char_limit=0,train_stroke_limit=0,
                          eval_every_epochs=2,decode_max_points=4,threads=2,final_test=True,checkpoint_every=1)
            config['model'].update(dim=16,max_points=8)
            path=tmp/'config.json';path.write_text(json.dumps(config));output=tmp/'run'
            cmd=[sys.executable,str(root/'models/scripts/train_single_stroke.py'),'--config',str(path),
                 '--output-dir',str(output),'--device','cpu']
            subprocess.run(cmd+['--max-steps','1'],cwd=root,check=True,capture_output=True,text=True)
            self.assertFalse(list(output.glob('eval_total_epoch_*')))
            self.assertFalse((output/'best.pt').exists())
            subprocess.run(cmd+['--resume',str(output/'last.pt'),'--max-steps','0'],cwd=root,check=True,capture_output=True,text=True)
            validations=[r for r in map(json.loads,(output/'metrics.jsonl').read_text().splitlines()) if r['kind']=='validation']
            self.assertEqual([r['epoch'] for r in validations],[2])
            self.assertEqual(validations[0]['strokes'],4)
            self.assertTrue(validations[0]['full_split'])
            usage=json.loads((output/'data_usage.json').read_text())
            gallery=json.loads((output/'eval_total_epoch_002/summary.json').read_text())
            self.assertEqual(usage['visualization_ids'],gallery['visualization_ids'])
            test=json.loads((output/'test_best/summary.json').read_text())
            self.assertEqual(test['split'],'test');self.assertEqual(test['strokes'],4)
            self.assertTrue(test['full_split']);self.assertFalse(test['test_used_for_model_selection'])
            self.assertEqual(test['checkpoint_epoch'],2)
            test_mtime=(output/'test_best/summary.json').stat().st_mtime_ns
            subprocess.run(cmd+['--resume',str(output/'last.pt'),'--max-steps','0'],cwd=root,check=True,capture_output=True,text=True)
            self.assertEqual((output/'test_best/summary.json').stat().st_mtime_ns,test_mtime)
            self.assertEqual(len(list(output.glob('eval_total_epoch_*'))),1)

    def test_resume_matches_uninterrupted_training(self):
        import subprocess
        import sys
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);db,split=fixture(tmp)
            config=json.loads((root/'models/configs/single_stroke_smoke.json').read_text())
            config.update(dataset=str(db),split_file=str(split),epochs=10,train_char_limit=0,
                          train_stroke_limit=0,eval_every_epochs=1,threads=2,augment=True,decode_max_points=4,final_test=False,
                          optimizer='adamw',cosine_steps=0,checkpoint_every=1,
                          early_stopping=dict(patience=1,min_delta=1.e6,min_epochs=2))
            config['model'].update(dim=16,max_points=8,dropout=.1)
            path=tmp/'config.json';path.write_text(json.dumps(config))
            def run(output,steps,resume=None):
                cmd=[sys.executable,str(root/'models/scripts/train_single_stroke.py'),'--config',str(path),
                     '--output-dir',str(output),'--device','cpu','--max-steps',str(steps)]
                if resume:cmd+=['--resume',str(resume)]
                subprocess.run(cmd,cwd=root,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            full=tmp/'full';partial=tmp/'partial'
            run(full,4);run(partial,1);run(partial,4,partial/'last.pt')
            a=torch.load(full/'last.pt',map_location='cpu');b=torch.load(partial/'last.pt',map_location='cpu')
            for key,value in a['model'].items():
                torch.testing.assert_close(value,b['model'][key],rtol=0,atol=0)
            self.assertEqual((a['step'],a['epoch'],a['cursor']),(b['step'],b['epoch'],b['cursor']))
            self.assertEqual(a['scheduler'],b['scheduler'])
            self.assertEqual(a['early_stopping'],b['early_stopping'])
            for output in (full,partial):
                status=json.loads((output/'training_status.json').read_text())
                self.assertEqual(status['reason'],'early_stopping')
                self.assertEqual(status['epoch'],2)
                self.assertEqual(status['early_stopping']['bad_epochs'],1)
                self.assertEqual(status['early_stopping_monitor'],'1-free_trajectory_iou')
            output=tmp/'export'
            subprocess.run([sys.executable,str(root/'models/scripts/evaluate_single_stroke.py'),
                            '--checkpoint',str(full/'last.pt'),'--output',str(output),'--split','val',
                            '--limit','1','--decode-max-points','4'],cwd=root,check=True,
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            report=json.loads((output/'summary.json').read_text())
            self.assertEqual(report['split'],'val')
            self.assertEqual(report['strokes'],1)
            self.assertTrue(json.loads((output/'strokes.jsonl').read_text().splitlines()[0])['first_xyz_exact'])
            self.assertTrue((output/'index.html').exists())
            self.assertEqual(len(list(output.rglob('*.npz'))),1)


if __name__=='__main__':unittest.main()
