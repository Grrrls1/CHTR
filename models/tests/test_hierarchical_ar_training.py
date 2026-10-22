"""Real LMDB integration, tail batches, strict resume, and evaluation artifacts."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import cv2
import lmdb
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from models.datasets.hierarchical_ar_dataset import TrajectoryDataset
from models.networks.hierarchical_ar import HierarchicalAR
from models.losses.hierarchical_ar_loss import three_task_loss,sample_lengths
from models.training.hierarchical_ar_setup import read_checkpoint
from models.scripts import train_hierarchical_ar as trainer
from models.scripts import evaluate_hierarchical_ar as evaluator

ROOT=Path(__file__).resolve().parents[1]


def make_dataset(path):
    path.mkdir()
    with lmdb.open(str(path),map_size=8*1024**2) as env:
        with env.begin(write=True) as txn:
            ids=[f'sample_{i}' for i in range(5)]
            txn.put(b'__ids__',json.dumps(ids).encode());txn.put(b'__ids_ready__',json.dumps(ids).encode())
            for i,sid in enumerate(ids):
                counts=[[2],[1,3],[3,2,1],[4],[2,2]][i]
                image=np.full((32,32),255,np.uint8);cv2.line(image,(3,3+i),(25,20+i),0,2)
                _,encoded=cv2.imencode('.png',image)
                points=np.zeros((sum(counts),6),np.float32)
                points[:,0]=np.linspace(3,25,len(points));points[:,1]=np.linspace(3+i,20+i,len(points));points[:,2]=2
                offsets=np.r_[0,np.cumsum(counts)]
                f=io.BytesIO();np.savez_compressed(f,points=points,stroke_offsets=offsets,image_size=np.array(32))
                row=dict(sample_id=sid,character='字',image_key=sid+'_png',trajectory_key=sid+'_npz')
                txn.put(sid.encode(),json.dumps(row).encode());txn.put(row['image_key'].encode(),encoded.tobytes());txn.put(row['trajectory_key'].encode(),f.getvalue())
    # Deliberately no mask assets, stroke types or stroke_type_map.json.


class HierarchicalARTrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def configuration(self,root):
        config=json.loads((ROOT/'configs/hierarchical_ar.json').read_text())
        config.update(dataset=str(root/'data'),output_dir=str(root/'run'),device='cpu',batch_size=3,epochs=2,
                      checkpoint_every=1,eval_every_epochs=2,eval_limit=1,validation_fraction=.2,threads=1)
        config['model']=dict(dim=16,heads=2,stroke_layers=1,point_layers=1,backbone='cnn',pretrained=False,
                             memory_size=2,max_strokes=3,max_points=6,dropout=0.)
        config['early_stopping']=dict(patience=0,min_delta=.001,min_epochs=1)
        return config

    def run_training(self,root,config,resume=None):
        config_path=root/'config.json';config_path.write_text(json.dumps(config))
        argv=['train_hierarchical_ar.py','--config',str(config_path)]
        if resume:argv+=['--resume',str(resume)]
        with patch.object(sys,'argv',argv),patch.object(trainer.signal,'signal'),contextlib.redirect_stdout(io.StringIO()):
            trainer.main()
        return read_checkpoint(Path(config['output_dir'])/'last.pt')

    def test_dataset_requires_no_auxiliary_annotations_and_validates_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_dataset(root/'data');ds=TrajectoryDataset(root/'data')
            try:
                lengths,digest=ds.inspect(3,6)
                self.assertEqual(len(lengths),5);self.assertEqual(len(digest),64)
                sample=ds[0]
                self.assertEqual(set(sample),{'image','points','stroke_offsets','image_size','sample_id','character'})
                self.assertEqual(sample['points'].shape[1],3)
                with self.assertRaisesRegex(ValueError,'no silent truncation'):ds.inspect(1,2)
            finally:ds.close()

    def test_tail_batch_and_mid_epoch_resume_match_uninterrupted_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_dataset(root/'data');base=self.configuration(root)
            full=self.run_training(root,base)
            partial=copy.deepcopy(base);partial.update(output_dir=str(root/'resumed'),max_steps=1)
            first=self.run_training(root,partial)
            self.assertEqual((first['cursor'],first['early_stopping']['samples']),(3,3))
            partial['max_steps']=0
            resumed=self.run_training(root,partial,root/'resumed/last.pt')
            self.assertEqual((full['step'],full['epoch'],full['cursor']),(resumed['step'],resumed['epoch'],resumed['cursor']))
            self.assertEqual(full['early_stopping'],resumed['early_stopping'])
            for name,value in full['model'].items():torch.testing.assert_close(value,resumed['model'][name],rtol=0,atol=0,msg=name)
            logs=[json.loads(l) for l in (root/'run/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['samples_in_update'] for r in logs],[3,1,3,1])
            split=json.loads((root/'run/split.json').read_text())
            self.assertFalse(set(split['train_ids'])&set(split['heldout_ids']))
            self.assertEqual((len(split['train_ids']),len(split['heldout_ids'])),(4,1))
            self.assertEqual(json.loads((root/'run/training_status.json').read_text())['reason'],'max_epochs')
            out=root/'run/eval_step_000004/heldout_images'
            self.assertTrue((out/'index.html').is_file());self.assertTrue((out/'comparison_sheet.png').is_file())
            summary=json.loads((out/'summary.json').read_text());row=summary['rows'][0]
            self.assertFalse(row['free_decode_uses_ground_truth']);self.assertTrue(row['diagnostic_uses_ground_truth_lengths'])
            with np.load(out/row['sample_id']/'prediction.npz',allow_pickle=False) as z:
                self.assertEqual(z['points'].shape[1],6);self.assertFalse(bool(z['used_ground_truth']))
                if not bool(z['completed']):self.assertFalse(bool(z['points'][-1,5]))
            argv=['evaluate_hierarchical_ar.py','--checkpoint',str(root/'run/last.pt'),'--output',str(root/'offline'),
                  '--device','cpu','--limit','1']
            with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):evaluator.main()
            self.assertTrue((root/'offline/summary.json').is_file())

    def test_accumulation_matches_parallel_batch_with_equal_image_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_dataset(root/'data');config=self.configuration(root)
            ds=TrajectoryDataset(root/'data');torch.manual_seed(42)
            a=HierarchicalAR(**config['model']).eval();b=copy.deepcopy(a)
            try:
                config['batch_size']=3
                la=trainer.train_update(a,ds,[0,1,2],'cpu',config)
                config['batch_size']=2
                lb=trainer.train_update(b,ds,[0,1,2],'cpu',config)
                self.assertAlmostEqual(la['loss'],lb['loss'],places=5)
                for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=3e-3,atol=2e-4,msg=name)
            finally:ds.close()

    def test_rejects_old_checkpoint_and_changed_resume_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_dataset(root/'data');config=self.configuration(root);config['max_steps']=1
            self.run_training(root,config)
            config['batch_size']=2
            with self.assertRaisesRegex(ValueError,'settings differ'):self.run_training(root,config,root/'run/last.pt')
            torch.save(dict(free_rollout_version=2,config={}),root/'legacy.pt')
            with self.assertRaisesRegex(ValueError,'legacy'):read_checkpoint(root/'legacy.pt')

    def test_resnet_geometry_only_updates_all_upstream_modules(self):
        config=json.loads((ROOT/'configs/hierarchical_ar.json').read_text())
        config['model'].update(dim=16,heads=2,stroke_layers=1,point_layers=1,memory_size=2,max_strokes=2,max_points=4)
        model=HierarchicalAR(**config['model']).train()
        images=torch.rand(2,1,64,64)
        rows=[dict(image=image,points=torch.rand(2,3),stroke_offsets=torch.tensor([0,2]),image_size=64) for image in images]
        predictions=model(images,[[2],[2]])
        _,terms=three_task_loss(predictions,rows);terms['geo'].backward()
        for module in (model.encoder.stages[-1],model.stroke_decoder,model.condition,model.geometry_head):
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads);self.assertGreater(sum(g.abs().sum().item() for g in grads),0.)
        for name,param in model.encoder.named_parameters():self.assertIsNotNone(param.grad,name)
        self.assertIsNone(model.point_end_head.weight.grad)
        self.assertIsNone(model.stroke_end_head.weight.grad)


if __name__=='__main__':unittest.main()
