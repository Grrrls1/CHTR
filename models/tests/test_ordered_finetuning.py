import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from models.networks.ordered_stroke import OrderedStrokeModel, MODE
from models.networks.single_stroke import SingleStrokeModel, MODE as POINT_MODE
from models.scripts.train_ordered_stroke import main
from models.tests.test_single_stroke import fixture
from models.training.single_stroke import initialize_finetuning


class OrderedFineTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_weights_only_expansion_and_cascade_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, split = fixture(root)
            old_spec = dict(dim=16, heads=4, encoder_layers=1, decoder_layers=2,
                            max_strokes=2, character_backbone='resnet18', history_chunk_size=2, dropout=0.)
            old = OrderedStrokeModel(**old_spec)
            parent = root/'parent.pt'
            torch.save(dict(training_mode=MODE, model=old.state_dict(),
                            config=dict(model=old_spec, dataset='CASIA'), epoch=6, step=999), parent)
            spec = dict(old_spec, max_strokes=4)
            config = dict(model=spec, init_checkpoint=str(parent), init_position_resize='copy_prefix',
                          init_checkpoint_sha256=hashlib.sha256(parent.read_bytes()).hexdigest())
            target = OrderedStrokeModel(**spec)
            added = target.position.weight.detach().clone()[3:]
            lineage = initialize_finetuning(target, config, training_mode=MODE, capacity_key='max_strokes')
            self.assertTrue(torch.equal(old.position.weight, target.position.weight[:3]))
            self.assertTrue(torch.equal(added, target.position.weight[3:]))
            self.assertTrue(torch.equal(old.valid_head.weight, target.valid_head.weight))
            self.assertEqual(lineage['position_migration']['new_max_strokes'], 4)
            with self.assertRaisesRegex(ValueError, 'different training stage'):
                initialize_finetuning(target, config)
            point_spec = dict(dim=16, heads=4, encoder_layers=1, decoder_layers=4, max_points=3, dropout=0.)
            point = root/'point.pt'
            point_model = SingleStrokeModel(**point_spec)
            point_state = {k: v.clone() for k, v in point_model.state_dict().items()}
            torch.save(dict(training_mode=POINT_MODE, model=point_state,
                            config=dict(model=point_spec, decode_max_points=3), epoch=13, step=123), point)
            config.update(training_mode=MODE, dataset=str(db), split_file=str(split),
                          dataset_backend='calligraphy_lmdb', output_dir=str(root/'train'),
                          seed=42, device='cpu', threads=2, batch_size=2, epochs=1, max_steps=0,
                          train_char_limit=0, eval_char_limit=0, learning_rate=.0001,
                          weight_decay=0., grad_clip=1., checkpoint_every=100, eval_every_epochs=1,
                          eval_batch_size=2, decode_max_strokes=4, validity_threshold=.5,
                          visualize_characters=1, loss_weights=dict(cls=4.,mask=5.,dice=5.,box=5.,giou=2.,start=5.,width=1.),
                          auxiliary_weight=1., point_checkpoint=str(point),
                          point_checkpoint_sha256=hashlib.sha256(point.read_bytes()).hexdigest(),
                          point_batch_size=2, point_decode_max_points=3, point_validity_threshold=.5,
                          selection_metric='casia_metrics.aiou')
            config_file = root/'config.json'
            config_file.write_text(json.dumps(config))
            with patch.object(sys, 'argv', ['train_ordered_stroke', '--config', str(config_file)]):
                main()
            ck = torch.load(root/'train/best.pt', map_location='cpu', weights_only=False)
            self.assertEqual((ck['epoch'], ck['step'], ck['cursor']), (1, 1, 0))
            self.assertEqual(ck['lineage']['parent_step'], 999)
            self.assertTrue(all(int(s['step']) == 1 for s in ck['optimizer']['state'].values()))
            report = json.loads((root/'train/eval_epoch_001/summary.json').read_text())
            self.assertTrue(report['full_split'])
            self.assertEqual(report['selection_metric'], 'casia_metrics.aiou')
            self.assertEqual(ck['best_selection_score'], report['casia_metrics']['aiou'])
            self.assertIn('character_trajectory_iou', report['point_layer'])
            saved_point = torch.load(point, map_location='cpu', weights_only=False)
            self.assertTrue(all(torch.equal(v, saved_point['model'][k]) for k, v in point_state.items()))


if __name__ == '__main__':
    unittest.main()
