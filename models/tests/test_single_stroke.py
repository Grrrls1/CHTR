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
            mask[25,25]=96;mask[26,26]=180  # Preserve decoded values, not a thresholded reconstruction.
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
                    self.assertEqual(float(s['image'][0,40,40]),0.)
                    self.assertAlmostEqual(float(s['image'][0,25,25]),96/255.)
                    self.assertAlmostEqual(float(s['image'][0,26,26]),180/255.)
                    sid,k=ds.key(1)
                    with catalog._env().begin() as txn:
                        raw=txn.get(catalog.rows[sid]['masks_key'].encode())
                    with np.load(io.BytesIO(raw)) as z:
                        stored=cv2.imdecode(z['stroke%03d.png'%k],cv2.IMREAD_GRAYSCALE)
                    np.testing.assert_array_equal(np.rint(s['image'][0].numpy()*255).astype(np.uint8),stored)
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

    def test_predicted_mask_adapter_is_separate(self):
        image=np.full((256,256),60,np.uint8);mask=np.zeros((256,256),np.float32)
        mask[20:40,20:40]=.8
        result=masked_stroke_image(image,mask)
        self.assertEqual(int(result[30,30]),60)
        self.assertEqual(int(result[50,50]),255)


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


class TrainingIntegrationTests(unittest.TestCase):
    def test_legacy_input_checkpoint_rejected(self):
        from models.networks.single_stroke import MODE
        from models.training.single_stroke import load_model
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'legacy.pt'
            torch.save(dict(training_mode=MODE),path)
            with self.assertRaisesRegex(ValueError,'input protocol'):
                load_model(path,'cpu')

    def test_resume_matches_uninterrupted_training(self):
        import subprocess
        import sys
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);db,split=fixture(tmp)
            config=json.loads((root/'models/configs/single_stroke_smoke.json').read_text())
            config.update(dataset=str(db),split_file=str(split),epochs=10,train_char_limit=0,
                          train_stroke_limit=0,eval_strokes=1,eval_every_steps=100,threads=2,augment=True)
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
            output=tmp/'export'
            subprocess.run([sys.executable,str(root/'models/scripts/evaluate_single_stroke.py'),
                            '--checkpoint',str(full/'last.pt'),'--output',str(output),'--split','val',
                            '--limit','1','--decode-max-points','4'],cwd=root,check=True,
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            report=json.loads((output/'summary.json').read_text())
            self.assertEqual(report['split'],'val')
            self.assertTrue(report['samples'][0]['first_xyz_exact'])
            self.assertTrue((output/'index.html').exists())
            self.assertEqual(len(list(output.glob('*.npz'))),1)


if __name__=='__main__':unittest.main()
