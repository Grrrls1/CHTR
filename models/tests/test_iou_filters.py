import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import lmdb

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from models.datasets.filtered_lmdb_dataset import CHTRFilteredDataset

MAP=ROOT.parent/'reproduction/pipeline_runs/mmah_single_path_gpu2_20261005_five/dataset_simplified_dataloader_v2/stroke_type_map.json'


class WholeIoUSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)
        shutil.copyfile(MAP,self.path/'stroke_type_map.json')
        metrics={
            'boundary':dict(trajectory_whole=.8,mask_cleanup_whole=.9,mask_extraction_whole=.9),
            'low_trajectory':dict(trajectory_whole=.7,mask_cleanup_whole=.95,mask_extraction_whole=.95),
            'low_cleaned_mask':dict(trajectory_whole=.95,mask_cleanup_whole=.85,mask_extraction_whole=.99),
            'missing':dict(mask_cleanup_whole=.99,mask_extraction_whole=.99),
            'nonfinite':dict(trajectory_whole=float('nan'),mask_cleanup_whole=.99,mask_extraction_whole=.99)}
        ids=list(metrics)+['pending']
        with lmdb.open(str(self.path),map_size=4*1024**2) as env:
            with env.begin(write=True) as txn:
                txn.put(b'__ids__',json.dumps(ids).encode())
                txn.put(b'__ids_ready__',json.dumps(list(metrics)).encode())
                txn.put(b'__meta__',json.dumps(dict(count=6,pending=1)).encode())
                for sid,values in metrics.items():txn.put(sid.encode(),json.dumps(dict(stage_iou=values,trajectory_key=sid+'/trajectory')).encode())
                txn.put(b'pending',json.dumps(dict(trajectory_key=None,stage_iou={})).encode())

    def tearDown(self):self.temp.cleanup()

    def test_no_quality_filter_by_default(self):
        ds=CHTRFilteredDataset(self.path)
        self.assertEqual(len(ds.all_ids),6);self.assertEqual(len(ds),5)
        self.assertEqual(ds.pending_count,1);self.assertEqual(ds.selection_summary['quality_excluded'],0)
        ds.close()

    def test_thresholds_are_inclusive_and_both_must_pass(self):
        ds=CHTRFilteredDataset(self.path,min_trajectory_iou=.8,min_mask_iou=.9)
        self.assertEqual(ds.ids,['boundary'])
        self.assertEqual(ds.selection_summary['rejected_trajectory_iou'],3)
        self.assertEqual(ds.selection_summary['rejected_mask_iou'],1)
        self.assertEqual(ds.selection_summary['selected'],1);ds.close()

    def test_mask_stage_is_explicit(self):
        ds=CHTRFilteredDataset(self.path,min_trajectory_iou=.8,min_mask_iou=.9,
                               mask_iou_stage='mask_extraction_whole')
        self.assertEqual(ds.ids,['boundary','low_cleaned_mask']);ds.close()

    def test_zero_threshold_does_not_disable_missing_value_checks(self):
        ds=CHTRFilteredDataset(self.path,min_trajectory_iou=0)
        self.assertEqual(ds.ids,['boundary','low_trajectory','low_cleaned_mask']);ds.close()

    def test_invalid_threshold_or_empty_selection_fails_clearly(self):
        for value in (-.1,1.1,float('nan'),True):
            with self.assertRaises(ValueError):CHTRFilteredDataset(self.path,min_mask_iou=value)
        with self.assertRaisesRegex(ValueError,'No samples remain'):
            CHTRFilteredDataset(self.path,min_trajectory_iou=1)
        with self.assertRaises(ValueError):CHTRFilteredDataset(self.path,mask_iou_stage='unknown')


if __name__=='__main__':unittest.main()
