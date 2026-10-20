import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

PIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE)); sys.path.insert(0, str(PIPE.parents[1] / 'data'))
from common import arrays, gray, iter_records, read_json, write_json, write_records
import skeleton_path_trajectory as baseline
from skeleton_gpu_batch import thin_batch, trajectory_from_skeleton
from continue_skeleton_dataset import (TYPE_MAP, eligible, pack_training_lmdb, skeleton_batch,
                                        training_selection, validate_training_row)
from models.datasets.lmdb_dataset import CHTRModelDataset, collate_strokes, load_type_map
from models.networks.model import StrokeTrajectoryModel
from models.losses.multitask import multitask_loss


class SkeletonContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        torch.set_num_threads(1)
        self.settings = dict(spacing=1.5,smooth_radius=2,min_branch_length=4.,supersample=4,
                             max_points=1024,iou_threshold=.87)

    def tearDown(self):
        self.temp.cleanup()

    def test_batched_thinning_equals_original_pixel_algorithm(self):
        masks = np.zeros((6,32,32),np.uint8)
        cv2.line(masks[0],(3,5),(28,25),1,7)
        cv2.rectangle(masks[1],(4,4),(27,27),1,-1)
        cv2.circle(masks[2],(16,16),11,1,5)
        masks[3,0:15,0:15] = 1  # border behavior must also match
        masks[4,16,16] = 1
        # Sixth mask is empty: batch processing must not contaminate it.
        with patch.object(baseline.cv2,'ximgproc',None,create=True):
            expected = np.stack([baseline.skeletonize_mask(m) for m in masks])
        actual = thin_batch(masks,'cpu')
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(thin_batch(actual,'cpu'),actual)

    @unittest.skipUnless(os.environ.get('CHTR_TEST_CUDA') == '1', 'CUDA parity requires a Slurm allocation')
    def test_cuda_matches_cpu(self):
        masks = np.zeros((3,64,64),np.uint8)
        for i in range(3):cv2.line(masks[i],(5,10+i),(55,40+i),1,9+i*2)
        np.testing.assert_array_equal(thin_batch(masks,'cuda'),thin_batch(masks,'cpu'))

    def test_point_cap_resamples_geometry_and_revalidates(self):
        mask = np.zeros((64,64),np.uint8);cv2.line(mask,(6,32),(57,32),1,7)
        skeleton = thin_batch(mask[None],'cpu')[0]
        points, offsets, _, diagnostics = trajectory_from_skeleton(mask,skeleton,None,
            dict(self.settings,max_points=12))
        self.assertEqual(len(points),12)
        self.assertTrue(diagnostics['resampled_for_limit'])
        self.assertEqual(offsets.tolist(),[0,12])
        self.assertTrue((points[:,2] > 0).all())
        baseline.validate_points(points,offsets,size=64)
        # A disconnected input must not be silently turned into a single stroke.
        mask[3,3] = 1
        with self.assertRaises(ValueError):trajectory_from_skeleton(mask,skeleton,None,self.settings)

    def sample(self):
        mask = np.zeros((256,256),np.uint8)
        cv2.line(mask,(35,128),(220,128),1,9)
        image = self.root/'image.png';cv2.imwrite(str(image),255-mask*255)
        npz = self.root/'cleaned.npz';np.savez_compressed(npz,extractnet_strokes=mask[None],stroke_labels=np.array([1],np.int64))
        report = self.root/'cleanup.json'
        write_json(report,dict(sample_id='fixture',cleanup_status='clean_candidate',before=dict(iou=1.),after=dict(iou=1.)))
        return dict(sample_id='fixture',character='一',image_path=str(image),source_image_path=str(image),
                    extracted_path=str(npz),cleanup_report_path=str(report),cleanup_status='clean_candidate')

    def test_real_batch_model_loader_lmdb_and_backward(self):
        row = self.sample(); refs=self.root/'refs.jsonl';write_records(refs,[])
        config=dict(pipeline_python=sys.executable,quality_threshold=.87,trajectory_style='test',copy_assets=True)
        args=argparse.Namespace(device='cpu',gpu_batch_size=4,cpu_workers=1,max_points=1024,stroke_type_map=str(TYPE_MAP))
        _,names=load_type_map(TYPE_MAP)
        manifest=skeleton_batch(self.root/'batch',[row],config,refs,args,names)
        ready=list(iter_records(manifest));self.assertEqual(len(ready),1)
        self.assertTrue(eligible(ready[0]))
        validate_training_row(ready[0],names,1024)
        z=arrays(ready[0]['trajectory_path'])
        np.testing.assert_array_equal(z['stroke_offsets'],z['subpath_offsets'])
        # Repeat the batch: completed stage must be reused without launching a worker.
        self.assertEqual(skeleton_batch(self.root/'batch',[row],config,refs,args,names),manifest)
        training=training_selection(manifest.parent,TYPE_MAP)
        lmdb_path=pack_training_lmdb(training,self.root/'chtr_mmah_lmdb',TYPE_MAP,map_gb=.05,transaction_size=1)
        self.assertEqual((lmdb_path/'stroke_type_map.json').read_bytes(),TYPE_MAP.read_bytes())
        self.assertEqual(pack_training_lmdb(training,lmdb_path,TYPE_MAP,map_gb=.05),lmdb_path)
        ds=CHTRModelDataset(lmdb_path)
        self.assertEqual(len(ds),1)
        sample=ds[0];batch=collate_strokes([sample])
        model=StrokeTrajectoryModel(dim=32,heads=4,stroke_layers=1,point_layers=1,dropout=0,memory_size=4)
        out=model(batch);loss,_=multitask_loss(out,batch)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
        ds.close()

    def test_worker_candidate_resume_does_not_recompute(self):
        row=self.sample();mask_path=self.root/'stroke.png';cv2.imwrite(str(mask_path),gray(row['image_path']))
        row.update(stroke_id='fixture_stroke000',stroke_index=0,stroke_label=1,input_png=str(mask_path),expected_strokes=1)
        manifest=self.root/'strokes.jsonl';write_records(manifest,[row]);refs=self.root/'refs.jsonl';write_records(refs,[])
        output=self.root/'output'
        command=[sys.executable,str(PIPE/'skeleton_gpu_batch.py'),'--manifest',str(manifest),
                 '--output',str(output),'--mmah-manifest',str(refs),'--device','cpu','--cpu-workers','1']
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        self.assertEqual(result.returncode,0,result.stdout)
        npz=output/'fixture_stroke000/trajectory.npz';mtime=npz.stat().st_mtime_ns
        result=subprocess.run(command+['--resume'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        self.assertEqual(result.returncode,0,result.stdout)
        self.assertEqual(read_json(output/'summary.json')['reused'],1)
        self.assertEqual(npz.stat().st_mtime_ns,mtime)
        # A changed image must invalidate the cached stroke, rather than silently reuse it.
        cv2.imwrite(str(mask_path),np.full((256,256),255,np.uint8))
        result=subprocess.run(command+['--resume'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Cached stroke inputs/config changed',result.stdout)


if __name__ == '__main__':unittest.main()
