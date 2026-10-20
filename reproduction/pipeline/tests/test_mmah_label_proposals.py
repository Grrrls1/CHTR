"""Guard against silent taxonomy guesses when proposing ordered stroke labels."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from propose_mmah_stroke_labels import propose


class ProposalTests(unittest.TestCase):
    def test_ambiguous_class_stays_unknown_without_anchor(self):
        official = dict(stroke_count=3)
        v2 = dict(stroke_count=3, category_status='complete', stroke_labels=[18, 6, 8])
        labels, statuses, anchor = propose(official, v2, None)
        self.assertEqual(labels, [0, -1, 1])
        self.assertEqual(statuses[1], 'ambiguous_dian_or_piedian')
        self.assertIsNone(anchor)

    def test_anchor_must_agree_on_every_unambiguous_stroke(self):
        official = dict(stroke_count=3)
        v2 = dict(stroke_count=3, category_status='complete', stroke_labels=[18, 6, 8])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'anchor.npz'
            np.savez_compressed(path, stroke_label=np.array([0, 22, 1]))
            labels, statuses, _ = propose(official, v2, dict(npz_path=str(path), source_split='train'))
            self.assertEqual(labels, [0, 22, 1])
            self.assertEqual(statuses[1], 'rhsedb_same_index_proposal')
            labels, _, _ = propose(official, v2, dict(npz_path=str(path), source_split='test'))
            self.assertEqual(labels, [0, -1, 1])
            np.savez_compressed(path, stroke_label=np.array([3, 22, 1]))
            labels, statuses, _ = propose(official, v2, dict(npz_path=str(path), source_split='train'))
            self.assertEqual(labels, [0, -1, 1])
            self.assertEqual(statuses[1], 'ambiguous_dian_or_piedian')

    def test_missing_source_never_receives_labels(self):
        labels, statuses, _ = propose(dict(stroke_count=2), None, None)
        self.assertEqual(labels, [-1, -1])
        self.assertEqual(statuses, ['no_ccse_category_source'] * 2)

    def test_user_equivalence_can_collapse_piedian_to_dian(self):
        official = dict(stroke_count=2)
        v2 = dict(stroke_count=2, category_status='complete', stroke_labels=[6, 11])
        labels, statuses, _ = propose(official, v2, None, 'collapse_to_dian')
        self.assertEqual(labels, [0, 13])
        self.assertEqual(statuses, ['user_equivalent_dian_piedian_proposal',
                                    'taxonomy_low_support_proposal'])


if __name__ == '__main__':
    unittest.main()
