import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch

from models.datasets.single_stroke import StrokeCatalog, masked_stroke_image
from models.datasets.ordered_stroke import OrderedStrokeDataset, collate_ordered_strokes
from models.losses.ordered_stroke import ordered_stroke_loss, aligned_giou
from models.networks.ordered_stroke import OrderedStrokeModel
from models.networks.two_stage import recover_trajectories
from models.tests.test_single_stroke import fixture


def small_model(**kwargs):
    settings = dict(dim=16, heads=4, encoder_layers=1, decoder_layers=2,
                    max_strokes=4, character_backbone='resnet18', dropout=0.)
    settings.update(kwargs)
    return OrderedStrokeModel(**settings)


class FirstPointOnly(torch.nn.Module):
    def generate(self,images,first_points,**kwargs):
        return [dict(points=p[None].clone(),terminated=True,truncated=False,stop_reason='invalid') for p in first_points]


class OrderedDataTests(unittest.TestCase):
    def test_character_labels_terminal_padding_and_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            db, split = fixture(Path(tmp))
            catalog = StrokeCatalog(db, split)
            try:
                dataset = OrderedStrokeDataset(catalog, 'train', 2)
                sample = dataset[0]
                self.assertEqual(len(dataset), len(catalog.split['train_ids']))
                torch.testing.assert_close(sample['starts'][0], torch.tensor([30/255, 30/255, 4/256]))
                torch.testing.assert_close(sample['boxes'][0], torch.tensor([20/255, 20/255, 79/255, 79/255]))
                short = dict(sample, masks=sample['masks'][:1], boxes=sample['boxes'][:1], starts=sample['starts'][:1])
                batch = collate_ordered_strokes([sample, short])
                self.assertEqual(batch['valid'].tolist(), [[True, True, False], [True, False, False]])
                self.assertEqual(batch['active'].tolist(), [[True, True, True], [True, True, False]])
                self.assertEqual(batch['history_active'].tolist(), [[True, True], [True, False]])
                self.assertTrue(torch.equal(batch['history_masks'][0], sample['masks']))
                with self.assertRaises(ValueError):
                    OrderedStrokeDataset(catalog, 'train', 1)
            finally:
                catalog.close()


class OrderedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(31)
        self.model = small_model()
        self.images = torch.rand(2, 1, 256, 256)
        self.masks = (torch.rand(2, 2, 256, 256) > .8).float()
        self.active = torch.tensor([[True, True], [True, False]])

    def test_heads_losses_gradients_and_no_terminal_geometry(self):
        prediction = self.model(self.images, self.masks, self.active)
        self.assertEqual(prediction['mask_logits'].shape, (2, 3, 64, 64))
        self.assertEqual(prediction['starts'].shape, (2, 3, 3))
        self.assertEqual(len(prediction['auxiliary']), 1)
        batch = dict(valid=torch.tensor([[True, True, False], [True, False, False]]),
                     active=torch.tensor([[True, True, True], [True, True, False]]),
                     masks=torch.rand(2, 3, 256, 256).round(), boxes=torch.tensor([.1, .2, .8, .9]).expand(2, 3, 4).clone(),
                     starts=torch.rand(2, 3, 3))
        loss, terms = ordered_stroke_loss(prediction, batch)
        loss.backward()
        for name in ('mask_head.weight', 'box_head.weight', 'start_xy_head.weight', 'start_width_head.weight',
                     'valid_head.weight', 'history_encoder.backbone.stem.0.weight', 'visual.lateral.0.weight'):
            grad = dict(self.model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.abs().sum()), 0., name)
        self.assertFalse(any('end_head' in name for name, _ in self.model.named_parameters()))
        changed = {k: v.clone() for k, v in batch.items()}
        for key in ('masks', 'boxes', 'starts'):
            changed[key][~batch['valid']] = float('nan')
        other, _ = ordered_stroke_loss(prediction, changed)
        torch.testing.assert_close(loss, other)
        torch.testing.assert_close(aligned_giou(batch['boxes'], batch['boxes']), torch.ones(2, 3))

    def test_train_mode_causality_shift_padding_and_chunk_invariance(self):
        # Crucially test TRAIN mode: packed-history BatchNorm would leak here.
        self.model.train()
        with torch.no_grad():
            a = self.model(self.images, self.masks, self.active)
            changed = self.masks.clone()
            changed[0, 1] = 1-changed[0, 1]
            changed[1, 1] = 123.
            b = self.model(self.images, changed, self.active)
            for key in ('starts', 'boxes', 'valid_logits', 'mask_logits'):
                torch.testing.assert_close(a[key][0, :2], b[key][0, :2], atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(a[key][1], b[key][1], atol=1e-6, rtol=1e-5)
            changed[:, 0] = 1-changed[:, 0]
            c = self.model(self.images, changed, self.active)
            torch.testing.assert_close(a['starts'][:, 0], c['starts'][:, 0])
            self.assertGreater(float((a['starts'][0, 1]-c['starts'][0, 1]).abs().sum()), 1e-6)
            self.model.history_chunk_size = 1
            d = self.model(self.images, self.masks, self.active)
            torch.testing.assert_close(a['starts'], d['starts'], atol=1e-6, rtol=1e-5)

    def test_image_only_stop_cap_and_batch_parity(self):
        self.model.eval()
        with torch.no_grad():
            self.model.valid_head.weight.zero_()
            self.model.valid_head.bias.fill_(-20)
        results = self.model.generate(self.images, max_strokes=2)
        self.assertTrue(all(r['terminated'] and len(r['starts']) == 0 for r in results))
        with torch.no_grad():
            self.model.valid_head.bias.fill_(20)
        results = self.model.generate(self.images, max_strokes=2)
        for i, result in enumerate(results):
            self.assertTrue(result['truncated'])
            self.assertEqual(len(result['starts']), 2)
            single = self.model.generate(self.images[i:i+1], max_strokes=2)[0]
            torch.testing.assert_close(result['masks'], single['masks'], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(result['starts'], single['starts'], atol=1e-6, rtol=1e-5)
        original = self.model.decode
        # BOS always valid; invalidate a subset at the second decision to exercise
        # removal of finished characters while other histories continue.
        def mixed(visual, history, active, auxiliary=False):
            out = original(visual, history, active, auxiliary)
            if len(history) == 2 and history.shape[1] == 1:
                out['valid_logits'][0, -1] = -20
            return out
        self.model.decode = mixed
        mixed_results = self.model.generate(self.images, max_strokes=2)
        self.assertEqual([len(r['starts']) for r in mixed_results], [1, 2])
        self.assertTrue(mixed_results[0]['terminated'])
        self.assertTrue(mixed_results[1]['truncated'])

    def test_production_swin_and_nine_decoder_layers(self):
        model = small_model(character_backbone='swin_t', decoder_layers=9).eval()
        with torch.no_grad():
            out = model(self.images[:1], self.masks[:1, :1], self.active[:1, :1])
        self.assertEqual(len(out['auxiliary']), 8)
        self.assertEqual(out['mask_logits'].shape, (1, 2, 64, 64))
        self.assertTrue(torch.isfinite(out['starts']).all())

    def test_small_fixed_batch_loss_decreases(self):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        targets = torch.zeros(2, 3, 256, 256)
        targets[:, :, 40:100, 40:150] = 1.
        batch = dict(valid=torch.tensor([[True, True, False], [True, False, False]]),
                     active=torch.tensor([[True, True, True], [True, True, False]]),
                     masks=targets,
                     boxes=torch.tensor([.1, .1, .9, .9]).expand(2, 3, 4),
                     starts=torch.tensor([.2, .3, .02]).expand(2, 3, 3))
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            prediction = self.model(self.images, self.masks, self.active)
            loss, _ = ordered_stroke_loss(prediction, batch)
            losses.append(float(loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            optimizer.step()
        self.assertLess(losses[-1], losses[0]*.9)


class AdapterTests(unittest.TestCase):
    def test_masked_original_xyw_and_empty_character(self):
        image = torch.arange(256).float()[None, None, None].expand(2, 1, 256, 256)/255
        mask = torch.zeros(1, 256, 256)
        mask[:, 20:80, 30:90] = .8
        start = torch.tensor([[.2, .3, .04]])
        class Stroke(torch.nn.Module):
            def generate(self, images, **kwargs):
                return [dict(masks=mask, starts=start, terminated=True, truncated=False),
                        dict(masks=mask[:0], starts=start[:0], terminated=True, truncated=False)]
        class Point(torch.nn.Module):
            def generate(self, images, first_points, **kwargs):
                self.images, self.starts = images.clone(), first_points.clone()
                return [dict(points=p[None], terminated=True, truncated=False) for p in first_points]
        point = Point().eval()
        result = recover_trajectories(Stroke().eval(), point, image, point_batch_size=1)
        expected = masked_stroke_image((image[0, 0].numpy()*255).round().astype('uint8'), mask[0].numpy())
        torch.testing.assert_close(point.images[0, 0], torch.from_numpy(expected).float()/255)
        torch.testing.assert_close(point.starts, start)
        torch.testing.assert_close(result[0]['strokes'][0]['points'][0], start[0])
        self.assertEqual(len(result[1]['strokes']), 0)

    def test_image_only_cli_with_both_real_models_and_checkpoints(self):
        from models.networks.single_stroke import SingleStrokeModel, MODE as POINT_MODE
        from models.networks.ordered_stroke import MODE
        torch.set_num_threads(2)
        settings = dict(dim=16, heads=4, encoder_layers=1, decoder_layers=2,
                        max_strokes=2, character_backbone='resnet18', dropout=0.)
        point_settings = dict(dim=16, heads=4, encoder_layers=1, decoder_layers=4, max_points=3, dropout=0.)
        stroke = OrderedStrokeModel(**settings)
        point = SingleStrokeModel(**point_settings)
        with torch.no_grad():
            stroke.valid_head.weight.zero_()
            stroke.valid_head.bias.fill_(20)
            point.valid_head.weight.zero_()
            point.valid_head.bias.fill_(-20)
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            torch.save(dict(training_mode=MODE, model=stroke.state_dict(), epoch=1, step=1,
                            config=dict(model=settings, validity_threshold=.5)), tmp/'stroke.pt')
            torch.save(dict(training_mode=POINT_MODE, model=point.state_dict(), epoch=1, step=1,
                            config=dict(model=point_settings, validity_threshold=.5)), tmp/'point.pt')
            cv2.imwrite(str(tmp/'image.png'), np.tile(np.arange(256, dtype=np.uint8), (256, 1)))
            cmd = [sys.executable, str(root/'models/scripts/infer_two_stage.py'),
                   '--stroke-checkpoint', str(tmp/'stroke.pt'), '--point-checkpoint', str(tmp/'point.pt'),
                   '--image', str(tmp/'image.png'), '--output', str(tmp/'output'), '--threads', '2']
            result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
            with np.load(tmp/'output/prediction.npz') as z:
                np.testing.assert_array_equal(z['stroke_offsets'], [0, 1, 2])
                np.testing.assert_allclose(z['points'], z['starts'], atol=1e-5)
                self.assertEqual(z['masks'].shape, (2, 256, 256))
            report = json.loads((tmp/'output/summary.json').read_text())
            self.assertEqual(report['stroke_stop_reason'], 'stroke_cap')
            self.assertEqual(report['point_stop_reasons'], ['invalid', 'invalid'])
            self.assertFalse(report['visualization'])
            self.assertEqual({p.name for p in (tmp/'output').iterdir()},{'prediction.npz','summary.json'})
            viz_cmd=cmd.copy()
            viz_cmd[viz_cmd.index('--output')+1]=str(tmp/'visualized')
            result=subprocess.run(viz_cmd+['--visualize'],cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            self.assertTrue((tmp/'visualized/index.html').is_file())
            with np.load(tmp/'output/prediction.npz') as raw,np.load(tmp/'visualized/prediction.npz') as visual:
                np.testing.assert_array_equal(raw['points'],visual['points'])
                np.testing.assert_array_equal(raw['stroke_offsets'],visual['stroke_offsets'])
            timing = report['timing_seconds']
            self.assertGreater(timing['inference'], 0)
            self.assertGreater(timing['model_load'], 0)
            self.assertGreaterEqual(timing['total_before_summary_write'], timing['inference'] + timing['model_load'])
            self.assertEqual(report['device'], 'cpu')


class EvaluationTests(unittest.TestCase):
    def test_order_count_penalty_and_zero_prediction(self):
        from models.training.ordered_stroke_evaluation import evaluate_split
        class Oracle(torch.nn.Module):
            def forward(self, images, masks, active):
                b, n = active.shape
                target = images.new_zeros(b, n+1, 256, 256)
                target[:, :n] = masks
                return dict(mask_logits=torch.where(target.bool(), 20., -20.),
                            starts=images.new_ones(b, n+1, 3)*.1,
                            boxes=images.new_tensor([.1, .1, .8, .8]).expand(b, n+1, 4),
                            valid_logits=torch.cat([active.float()*40-20, images.new_full((b, 1), -20.)], 1))
            def generate(self, images, cap, threshold):
                result = []
                for i in range(len(images)):
                    masks = images.new_zeros(1 if i == 0 else 0, 256, 256)
                    masks[:, 20:80, 20:80] = 1.
                    result.append(dict(masks=masks, starts=images.new_ones(len(masks), 3)*.1,
                                       boxes=images.new_zeros(len(masks), 4), confidences=images.new_ones(len(masks)),
                                       terminated=True, truncated=False, stop_reason='invalid'))
                return result
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db, split = fixture(tmp)
            catalog = StrokeCatalog(db, split)
            try:
                dataset = OrderedStrokeDataset(catalog, 'val', 4)
                config = dict(eval_batch_size=2, decode_max_strokes=4, validity_threshold=.5, visualize_characters=2,
                              point_batch_size=2,point_decode_max_points=4,point_validity_threshold=.5)
                model = Oracle().train()
                report = evaluate_split(model, dataset, 'cpu', config, tmp/'eval',point_model=FirstPointOnly().eval())
                self.assertTrue(model.training)
                # Two true strokes per character. First predicts one perfect
                # mask (IoU = 1/2), second predicts none (IoU = 0): mean 1/4.
                self.assertAlmostEqual(report['free']['ordered_mask_iou'], .25)
                self.assertEqual(report['free']['strict_sequence_mask50'], 0.)
                self.assertEqual(report['free']['stroke_count_mae'], 1.5)
                self.assertEqual(report['free']['aligned_strokes'], 1)
                self.assertEqual(report['teacher_forced']['mask_iou'], 1.)
                self.assertEqual(report['teacher_forced']['terminal_recall'], 1.)
                self.assertTrue(report['full_split'])
                self.assertTrue((tmp/'eval/index.html').is_file())
                self.assertEqual(report['point_layer']['missing_strokes'],3)
                self.assertEqual(report['point_layer']['predicted_strokes'],1)
                self.assertIn('stroke_layer',report)
            finally:
                catalog.close()


class TrainingTests(unittest.TestCase):
    def test_resume_full_validation_and_independent_evaluation(self):
        torch.set_num_threads(2)
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db, split = fixture(tmp)
            config = json.loads((root/'models/configs/ordered_stroke_smoke.json').read_text())
            config.update(dataset=str(db), split_file=str(split), train_char_limit=0, eval_char_limit=0,
                          epochs=2, batch_size=1, decode_max_strokes=3, visualize_characters=2)
            config['model'].update(dim=16, max_strokes=4)
            from models.networks.single_stroke import SingleStrokeModel,MODE as POINT_MODE
            point_settings=dict(dim=16,heads=4,encoder_layers=1,decoder_layers=4,max_points=4,dropout=0.)
            point=SingleStrokeModel(**point_settings)
            torch.save(dict(training_mode=POINT_MODE,model=point.state_dict(),epoch=20,step=120100,
                            config=dict(model=point_settings,decode_max_points=4,validity_threshold=.5)),tmp/'point.pt')
            config.update(point_checkpoint=str(tmp/'point.pt'),point_decode_max_points=4,point_batch_size=2)
            path = tmp/'config.json'
            path.write_text(json.dumps(config))
            def run(output, steps, resume=None):
                cmd = [sys.executable, str(root/'models/scripts/train_ordered_stroke.py'), '--config', str(path),
                       '--output-dir', str(output), '--device', 'cpu', '--max-steps', str(steps)]
                if resume:
                    cmd += ['--resume', str(resume)]
                result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
            full, resumed = tmp/'full', tmp/'resumed'
            run(full, 4)
            run(resumed, 1)
            self.assertFalse((resumed/'best.pt').exists())
            run(resumed, 4, resumed/'last.pt')
            a = torch.load(full/'last.pt', map_location='cpu', weights_only=False)
            b = torch.load(resumed/'last.pt', map_location='cpu', weights_only=False)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            self.assertEqual((a['step'], a['epoch'], a['cursor']), (b['step'], b['epoch'], b['cursor']))
            del a, b
            summary = json.loads((full/'eval_epoch_002/summary.json').read_text())
            self.assertTrue(summary['full_split'])
            self.assertEqual(summary['strokes'], 4)
            self.assertFalse(summary['test_used_for_model_selection'])
            self.assertEqual(len(list((full/'eval_epoch_002').rglob('*.png'))),0)
            self.assertEqual(len(list((full/'eval_epoch_002').rglob('character_trajectory.npz'))),2)
            self.assertIn('point_layer',summary)
            self.assertEqual(summary['point_checkpoint']['epoch'],20)
            cmd = [sys.executable, str(root/'models/scripts/evaluate_ordered_stroke.py'),
                   '--checkpoint', str(full/'best.pt'), '--output', str(tmp/'test'), '--split', 'test', '--limit', '1']
            result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
            summary = json.loads((tmp/'test/summary.json').read_text())
            self.assertEqual(summary['split'], 'test')
            self.assertFalse(summary['full_split'])
            self.assertEqual(summary['characters'], 1)
            self.assertIn('epoch', summary['checkpoint'])


if __name__ == '__main__':
    unittest.main()
