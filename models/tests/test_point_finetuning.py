import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from models.networks.single_stroke import MODE, SingleStrokeModel
from models.scripts.train_single_stroke import main
from models.tests.test_single_stroke import fixture
from models.training.single_stroke import initialize_finetuning


class PointFineTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def parent(self, folder):
        spec = dict(dim=16, heads=4, encoder_layers=1, decoder_layers=4,
                    max_points=3, dropout=0.)
        model = SingleStrokeModel(**spec)
        path = folder/'parent.pt'
        # Deliberately incompatible historical progress must never enter a new run.
        torch.save(dict(training_mode=MODE, config=dict(model=spec, dataset='CASIA',
                        dataset_backend='casia_pot'), model=model.state_dict(),
                        epoch=99, step=9000, cursor=123, signature='parent-signature',
                        optimizer={'must_not_be_loaded': True}), path)
        return model, dict(model=dict(spec, max_points=4), init_checkpoint=str(path),
                           init_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           init_position_resize='copy_prefix')

    def test_preserves_pretrained_weights_and_new_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, config = self.parent(Path(tmp))
            target = SingleStrokeModel(**config['model'])
            new_positions = target.position.weight.detach().clone()[4:]
            lineage = initialize_finetuning(target, config)
            for key, value in parent.state_dict().items():
                actual = target.state_dict()[key]
                self.assertTrue(torch.equal(value, actual[:4] if key=='position.weight' else actual), key)
            self.assertTrue(torch.equal(new_positions, target.position.weight[4:]))
            self.assertFalse(lineage['optimizer_inherited'])
            self.assertEqual(lineage['position_migration']['new_rows'], 1)
            config['init_checkpoint_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                initialize_finetuning(target, config)

    def test_rejects_unapproved_architecture_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, config = self.parent(Path(tmp))
            target = SingleStrokeModel(**config['model'])
            del config['init_position_resize']
            with self.assertRaisesRegex(ValueError, 'architecture mismatch'):
                initialize_finetuning(target, config)

    def test_new_training_run_and_aiou_evaluation_on_calligraphy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, split = fixture(root)
            _, config = self.parent(root)
            config.update(training_mode=MODE, dataset=str(db), split_file=str(split),
                          dataset_backend='calligraphy_lmdb', output_dir=str(root/'train'),
                          seed=42, device='cpu', threads=2, batch_size=4, epochs=1,
                          max_steps=0, train_char_limit=0, train_stroke_limit=0,
                          augment=False, learning_rate=.0003, weight_decay=0.,
                          width_weight=1., valid_weight=.2, grad_clip=1., optimizer='adamw',
                          cosine_steps=0, checkpoint_every=100, eval_every_epochs=1,
                          evaluation_protocol='full_split_free_v2', eval_batch_size=2,
                          visualize_characters=1, decode_max_points=4, validity_threshold=.5,
                          selection_metric='casia_metrics.aiou', final_test=False,
                          early_stopping=dict(patience=10, min_delta=.001, min_epochs=20))
            config_file = root/'config.json'
            config_file.write_text(json.dumps(config))
            with patch.object(sys, 'argv', ['train_single_stroke', '--config', str(config_file)]):
                main()
            ck = torch.load(root/'train/last.pt', map_location='cpu', weights_only=False)
            self.assertEqual((ck['epoch'], ck['step'], ck['cursor']), (1, 1, 0))
            self.assertEqual(ck['lineage']['parent_step'], 9000)
            self.assertTrue(all(int(s['step']) == 1 for s in ck['optimizer']['state'].values()))
            report = json.loads((root/'train/eval_total_epoch_001/summary.json').read_text())
            self.assertTrue(report['full_split'])
            self.assertIn('aiou', report['casia_metrics'])
            self.assertIn('ldtw_px', report['casia_metrics'])
            self.assertFalse(report['render_protocol']['synthetic_width'])
            self.assertEqual(ck['best_selection_score'], report['casia_metrics']['aiou'])
            self.assertTrue((root/'train/fine_tuning.json').exists())
            self.assertTrue((root/'train/best.pt').exists())


if __name__ == '__main__':
    unittest.main()
