import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.training.early_stopping import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    def test_patience_delta_and_minimum_epochs(self):
        stop = EarlyStopping(dict(patience=2, min_delta=.1, min_epochs=4))
        results = []
        for epoch, loss in enumerate([2., 1.95, 1.94, 1.93], 1):
            stop.add(loss, 2)
            results.append(stop.finish_epoch(epoch)[2])
        self.assertEqual(results, [False, False, False, True])
        stop.add(1.5, 2)
        _, improved, ended = stop.finish_epoch(5)
        self.assertTrue(improved)
        self.assertFalse(ended)

    def test_partial_epoch_resume_and_unequal_batches(self):
        original = EarlyStopping(dict(patience=2))
        original.add(3., 2)
        restored = EarlyStopping(dict(patience=2))
        restored.load_state_dict(original.state_dict())
        for stop in (original, restored):
            stop.add(6., 1)
            mean, _, _ = stop.finish_epoch(1)
            self.assertEqual(mean, 4.)
        self.assertEqual(original.state_dict(), restored.state_dict())

    def test_disabled_and_invalid(self):
        stop = EarlyStopping()
        for epoch in range(1, 5):
            stop.add(2., 1)
            self.assertFalse(stop.finish_epoch(epoch)[2])
        for config in (dict(patience=-1), dict(patience=1.5), dict(min_delta=float('nan')),
                       dict(min_epochs=0)):
            with self.assertRaises(ValueError):
                EarlyStopping(config)


if __name__ == '__main__':
    unittest.main()
