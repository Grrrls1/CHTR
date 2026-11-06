import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from models.tests.test_casia_evaluation import pot_record
from models.datasets.casia_training import CasiaCatalog, render_paths, DEFAULT_RENDER
from models.datasets.single_stroke import SingleStrokeDataset, collate_single_strokes
from models.datasets.ordered_stroke import OrderedStrokeDataset, collate_ordered_strokes
from models.scripts.prepare_casia_training import prepare
from models.training.casia_benchmark_metrics import adaptive_iou, centerline_mask, trajectory_metrics
from models.training.single_stroke import make_data, signature


def fixture(root):
    for split, writers in [('train',['001','002','003']),('test',['004'])]:
        folder=root/'raw/1.1'/split
        folder.mkdir(parents=True)
        for writer in writers:
            (folder/(writer+'.pot')).write_bytes(
                pot_record('一', [[(0,0),(0,0),(30,0)],[(15,0)]])+pot_record('A',[[(0,0),(1,1)]]))
    prepare(root/'raw', root/'prepared', val_writers=1)
    return CasiaCatalog(root/'raw', root/'prepared/split.json')


class CasiaTrainingTests(unittest.TestCase):
    def test_two_stage_evaluation_with_nested_pot_ids(self):
        import torch
        from models.networks.single_stroke import SingleStrokeModel
        from models.networks.ordered_stroke import OrderedStrokeModel
        from models.training.single_stroke_evaluation import evaluate_split as point_eval
        from models.training.ordered_stroke_evaluation import evaluate_split as stroke_eval
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); c=fixture(root)
            try:
                point=SingleStrokeModel(dim=16,heads=4,encoder_layers=1,decoder_layers=4,max_points=3,dropout=0.)
                point.eval();point.requires_grad_(False)
                config=dict(dataset_backend='casia_pot',model=dict(max_points=3),width_weight=1.,valid_weight=.2,
                            eval_batch_size=2,decode_max_points=3,validity_threshold=.5,visualize_characters=1)
                report=point_eval(point,SingleStrokeDataset(c,'val',3),'cpu',config,root/'point')
                self.assertIn('aiou',report['casia_metrics'])
                self.assertTrue((root/'point'/c.split['val_ids'][0]/'index.html').exists())
                stroke=OrderedStrokeModel(dim=16,heads=4,encoder_layers=1,decoder_layers=2,max_strokes=2,
                                          character_backbone='resnet18',dropout=0.)
                config.update(decode_max_strokes=2,point_decode_max_points=3,point_validity_threshold=.5,point_batch_size=2)
                report=stroke_eval(stroke,OrderedStrokeDataset(c,'val',2),'cpu',config,root/'stroke',point_model=point)
                self.assertIn('aiou',report['casia_metrics'])
                self.assertEqual(report['selection_metric'],'casia_metrics.aiou')
            finally:c.close()

    def test_shared_assets_deterministic_width_and_no_label_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            c=fixture(Path(folder))
            try:
                point=SingleStrokeDataset(c,'train',3)
                stroke=OrderedStrokeDataset(c,'train',2)
                sid=point.ids[0]
                for epoch in (0,1,7):
                    point.render_epoch=stroke.render_epoch=epoch
                    a,b=point[0],stroke[0]
                    np.testing.assert_array_equal(a['points'][0],b['starts'][0])
                    self.assertEqual(len(a['points']),3)  # duplicates retained
                    self.assertEqual(float(a['points'][0,2]*256),c.width_for(sid,epoch))
                    img=np.rint(b['image'][0].numpy()*255).astype(np.uint8)
                    mask=b['masks'][0].numpy().astype(bool)
                    np.testing.assert_array_equal(a['image'][0].numpy()*255,np.where(mask,img,255))
                    np.testing.assert_array_equal(a['points'][0,:2],a['points'][1,:2])
                fixed=c.character_assets(sid)
                self.assertTrue((fixed[1][:,2]==8).all())
                widths={c.width_for(sid,e) for e in range(50)}
                self.assertGreater(len(widths),5)
                self.assertTrue(widths<=set(range(2,13)))
                with self.assertRaises(ValueError):
                    c.render_assets(c.split['val_ids'][0],'val',1)
                with self.assertRaisesRegex(ValueError,'no truncation'):
                    SingleStrokeDataset(c,'train',2)
                with self.assertRaisesRegex(ValueError,'no truncation'):
                    OrderedStrokeDataset(c,'train',1)
                batch=collate_single_strokes([point[0],point[1]])
                self.assertEqual(batch['lengths'].tolist(),[3,1])
                self.assertFalse(bool(batch['valid'][1].any()))
                batch=collate_ordered_strokes([stroke[0]])
                self.assertEqual(batch['valid'].tolist(),[[True,True,False]])
                self.assertEqual(batch['active'].tolist(),[[True,True,True]])
            finally:c.close()

    def test_fingerprints_backend_and_fixed_width_ablation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); c=fixture(root); c.close()
            config=dict(dataset_backend='casia_pot',dataset=str(root/'raw'),split_file=str(root/'prepared/split.json'),
                        model=dict(max_points=3),augment=False,train_char_limit=0,train_stroke_limit=0)
            c,train,val,indices,_=make_data(config)
            try:
                sig=signature(config,c)
                fixed=dict(config,render=dict(train_width_min=8,train_width_max=8))
                self.assertNotEqual(sig,signature(fixed,c))
                writers=[{c.rows[s]['writer_id'] for s in c.split[n+'_ids']} for n in ('train','val','test')]
                self.assertFalse(writers[0]&writers[1] or writers[0]&writers[2] or writers[1]&writers[2])
                self.assertEqual(len(indices),len(train))
                self.assertIsNone(val.render_epoch)
            finally:c.close()
            limited=dict(config,train_char_limit=1,train_stroke_limit=1,eval_char_limit=1)
            limited_catalog,_,_,train_indices,val_indices=make_data(limited)
            try:
                self.assertEqual(len(train_indices),1)
                self.assertEqual(len(val_indices),2)
            finally:limited_catalog.close()
            raw=next((root/'raw/1.1/train').glob('*.pot'))
            raw.write_bytes(raw.read_bytes()+b'bad')
            with self.assertRaisesRegex(ValueError,'source fingerprint'):
                CasiaCatalog(root/'raw',root/'prepared/split.json')

    def test_aiou_matches_explicit_dilation_and_order_metric(self):
        gt=[np.array([[20.,20.],[40.,20.],[40.,40.]])]
        image,_=render_paths(gt,8,DEFAULT_RENDER)
        pred=[gt[0]+[2,1]]
        result=adaptive_iou(image,pred)
        _,mask=cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        mask=mask.astype(bool)
        p=centerline_mask(pred).astype(np.uint8)
        scores=[]
        for _ in range(256):
            b=p.astype(bool)
            scores.append((mask&b).sum()/max(1,(mask|b).sum()))
            p=cv2.dilate(p,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        self.assertAlmostEqual(result['aiou'],max(scores),places=12)
        self.assertEqual(result['dilation_iterations'],int(np.argmax(scores)))
        self.assertEqual(adaptive_iou(image,[])['aiou'],0)
        same=trajectory_metrics(image,gt,gt)
        reverse=trajectory_metrics(image,gt,[gt[0][::-1]])
        self.assertEqual(same['aiou'],reverse['aiou'])
        self.assertEqual(same['ldtw_px'],0)
        self.assertGreater(reverse['ldtw_px'],0)
        self.assertIsNone(trajectory_metrics(image,gt,[])['ldtw_px'])


if __name__=='__main__':unittest.main()
