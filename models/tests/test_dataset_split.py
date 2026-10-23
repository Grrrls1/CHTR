import copy
import tempfile
import unittest
from pathlib import Path
import json

from models.datasets.dataset_split import create_split, validate_split, load_split_ids


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(sample_id=str(i), character='永' if i < 10 else '一', stroke_count=3)
                     for i in range(12)]
        self.groups = [[str(i)] for i in range(12)]

    def test_reproducible_coverage_and_rare(self):
        a = create_split(self.rows,self.groups)
        self.assertEqual(a,create_split(list(reversed(self.rows)),self.groups))
        self.assertTrue(all(a['characters']['永']['samples'].values()))
        self.assertEqual(a['characters']['一']['samples'],dict(train=2,val=0,test=0))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'split.json'; p.write_text(json.dumps(a))
            self.assertEqual(load_split_ids(p,self.rows,'val'),a['val_ids'])

    def test_duplicate_group_and_tampering(self):
        groups = [['0','1']] + [[str(i)] for i in range(2,12)]
        a = create_split(self.rows,groups)
        self.assertTrue(any({'0','1'} <= set(a[n+'_ids']) for n in ('train','val','test')))
        bad = copy.deepcopy(a); bad['val_ids'].append(bad['train_ids'][0])
        with self.assertRaises(ValueError): validate_split(bad,self.rows)
        bad = copy.deepcopy(a); bad['test_ids'] = ['unknown']
        with self.assertRaises(ValueError): validate_split(bad,self.rows)
        changed = copy.deepcopy(self.rows); changed[0]['stroke_count']=7
        with self.assertRaises(ValueError): validate_split(a,changed)

    def test_impossible_and_cross_character(self):
        with self.assertRaises(ValueError): create_split(self.rows,[list(map(str,range(12)))])
        groups = [['0','10']] + [[str(i)] for i in range(12) if i not in (0,10)]
        a = create_split(self.rows,groups)
        self.assertTrue({'0','10','11'} <= set(a['train_ids']))
        self.assertTrue(all(a['characters']['永']['samples'].values()))


class GroupingTests(unittest.TestCase):
    def test_hash_tree_matches_brute_force(self):
        import random
        from models.scripts.create_dataset_split import HashTree
        rng = random.Random(9); tree = HashTree(); values = []
        for i in range(100):
            value = rng.getrandbits(12)
            expected = {j for j,h in enumerate(values) if bin(h ^ value).count("1") <= 2}
            # Equal hashes share one representative in the tree.
            representative = {values.index(values[j]) for j in expected}
            self.assertEqual(set(tree.insert(value,i,2)),representative)
            values.append(value)

    def test_exact_pixels_grouped_from_lmdb(self):
        import io
        import lmdb
        import numpy as np
        from PIL import Image
        from models.scripts.create_dataset_split import group_candidates
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            pixels = np.random.default_rng(8).integers(0,255,(32,32),dtype=np.uint8)
            with lmdb.open(tmp,map_size=2**20) as env:
                with env.begin(write=True) as txn:
                    for i in range(3):
                        data = io.BytesIO()
                        Image.fromarray(pixels if i<2 else 255-pixels).save(data,format='PNG')
                        txn.put(str(i).encode(),data.getvalue())
                        rows.append(dict(sample_id=str(i),image_key=str(i)))
            groups,_ = group_candidates(tmp,rows,0)
            self.assertIn(['0','1'],groups)


if __name__ == '__main__': unittest.main()
