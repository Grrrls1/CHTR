"""Random initialization and checkpointable epoch-loss stopping for rollout training."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from models.training.free_rollout_setup import create_model
from models.scripts import train_free_rollout as trainer


MODEL = dict(dim=16, heads=2, stroke_layers=1, point_layers=1, dropout=0.,
             memory_size=2, max_strokes=3, max_points=6, separate_branches=True)


class FakeDataset:
    def __init__(self, path):
        self.ids = ['a', 'b', 'c']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return dict(image=torch.zeros(1, 32, 32), stroke_types=torch.tensor([1]))

    def _env(self):
        return self

    def begin(self):
        return contextlib.nullcontext(self)

    def get(self, key):
        return b'{"stroke_count": 1}'

    def close(self):
        pass


class RolloutTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_random_initialization_never_reads_checkpoint_and_uses_configured_dim(self):
        config = dict(model=dict(MODEL, dim=128, heads=4))
        with patch('torch.load', side_effect=AssertionError('unexpected checkpoint read')):
            torch.manual_seed(42)
            a, ck = create_model(config, 'cpu')
            torch.manual_seed(42)
            b, _ = create_model(config, 'cpu')
        self.assertIsNone(ck)
        self.assertEqual(a.dim, 128)
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)

    def test_explicit_checkpoint_requires_same_architecture(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'last.pt'
            a, _ = create_model(dict(model=MODEL), 'cpu')
            torch.save(dict(model=a.state_dict(), config=dict(model=MODEL)), path)
            with self.assertRaisesRegex(ValueError, 'architecture differs'):
                create_model(dict(model=dict(MODEL, dim=32)), 'cpu', path)

    def run_training(self, path, output, max_steps=0, resume=None, batch_size=None):
        config = dict(training_mode='free_rollout', dataset=str(path/'fake_data'), output_dir=str(output),
            model=MODEL, device='cpu', seed=42, threads=1, epochs=8, max_steps=max_steps,
            gradient_accumulation=2, checkpoint_every=1, train_limit=0, validation_limit=1,
            learning_rate=1e-5, weight_decay=0., grad_clip=1., rollout=dict(max_strokes=3,max_points=6),
            eval_every_epochs=20, require_full_dataset=True,
            early_stopping=dict(patience=1,min_delta=.001,min_epochs=2))
        if batch_size is not None:
            config.update(batch_size=batch_size, gradient_accumulation=1)
        config_path = path/'config.json'
        config_path.write_text(json.dumps(config))
        argv = ['train_free_rollout.py', '--config', str(config_path)]
        if resume is not None:
            argv += ['--resume', str(resume)]

        def fake_rollout(model, *args, **kwargs):
            return dict(strokes=[dict(geometry=model.point_bos[0, :, :3])],
                        completed=True, truncated=False)

        def point_loss(result, sample, config):
            zero = result['strokes'][0]['geometry'].sum()*0
            terms = {k: zero for k in ('xy','width','start','delta','state','length','missing','extra')}
            terms['xy'] = zero + 1
            return terms

        def fake_visuals(model, *args):
            return model.point_bos.sum()*0

        def visual_loss(zero, sample):
            return {k: zero for k in ('mask_bce','mask_dice','type','last','endpoints')}

        evaluation = dict(mean_image_only_iou=.1, rows=[])
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', argv))
            for name, value in [('CHTRModelDataset',FakeDataset), ('rollout',fake_rollout),
                ('rollout_batch',lambda model, images, **kw: [fake_rollout(model) for _ in images]),
                ('visual_predictions_batch',lambda model, images, counts: [fake_visuals(model) for _ in images]),
                ('trajectory_loss',point_loss), ('visual_predictions',fake_visuals), ('visual_loss',visual_loss)]:
                stack.enter_context(patch.object(trainer, name, value))
            stack.enter_context(patch.object(trainer, 'evaluate', return_value=evaluation.copy()))
            stack.enter_context(patch.object(trainer.subprocess, 'run'))
            stack.enter_context(patch.object(trainer.signal, 'signal'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            trainer.main()
        return torch.load(output/'last.pt', map_location='cpu', weights_only=False)

    def test_early_stopping_uses_epoch_loss_and_saves_best_loss_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);output=root/'train'
            ck = self.run_training(root, output)
            status = json.loads((output/'training_status.json').read_text())
            self.assertEqual(status['reason'], 'early_stopping')
            self.assertEqual((ck['epoch'], ck['step']), (2,4))
            self.assertEqual(ck['early_stopping']['best'], 1.)
            self.assertEqual(ck['early_stopping']['bad_epochs'], 1)
            self.assertEqual(ck['config']['initialization'], 'random')
            self.assertNotIn('initial_weights', ck['config'])
            self.assertTrue((output/'best_loss.pt').is_file())

    def test_partial_epoch_resume_keeps_loss_statistics_and_matches_continuous(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            full = self.run_training(root, root/'full')
            first = self.run_training(root, root/'resumed', max_steps=1)
            self.assertEqual(first['early_stopping']['samples'],2)
            self.assertEqual(first['early_stopping']['loss_sum'],2.)
            last = self.run_training(root, root/'resumed', resume=root/'resumed/last.pt')
            self.assertEqual(full['early_stopping'], last['early_stopping'])
            self.assertEqual((full['step'],full['epoch'],full['cursor']),
                             (last['step'],last['epoch'],last['cursor']))
            for name, value in full['model'].items():
                torch.testing.assert_close(value,last['model'][name],rtol=0,atol=0)

    def test_default_configuration_keeps_job237_stopping_rules(self):
        current=json.loads((ROOT/'configs/free_rollout.json').read_text())
        old=json.loads((ROOT/'configs/full_separated_10gb.json').read_text())
        self.assertEqual(current['early_stopping'],old['early_stopping'])
        self.assertEqual(current['model']['dim'],128)
        self.assertEqual(current['epochs'],500)
        self.assertEqual(current['dataset'],'../data/chtr_mmah_lmdb_1915')
        self.assertNotIn('initial_weights',current)
        self.assertEqual(current['batch_size'],24)

    def test_batched_tail_and_mid_epoch_resume_preserve_sample_accounting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            full = self.run_training(root, root/'full', batch_size=2)
            first = self.run_training(root, root/'resumed', max_steps=1, batch_size=2)
            self.assertEqual(first['cursor'],2)
            self.assertEqual(first['early_stopping']['samples'],2)
            resumed = self.run_training(root, root/'resumed', resume=root/'resumed/last.pt', batch_size=2)
            self.assertEqual(full['early_stopping'],resumed['early_stopping'])
            self.assertEqual((full['step'],full['epoch'],full['cursor']),
                             (resumed['step'],resumed['epoch'],resumed['cursor']))
            metrics = [json.loads(line) for line in (root/'full/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['samples_in_update'] for r in metrics],[2,1,2,1])
            self.assertTrue(all(r['loss']==1. for r in metrics))


if __name__ == '__main__':
    unittest.main()
