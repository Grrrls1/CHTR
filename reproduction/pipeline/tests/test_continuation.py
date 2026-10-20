import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

PIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE))
sys.path.insert(0, str(PIPE.parents[1] / 'data'))
from common import iter_records, preview, read_json, write_json, write_records
from continue_chtr_dataset import (checked_output, chunks, cleanup_summary, exclusive_run,
                                   merge_datasets, run_batch, stage)
from build_chtr_dataset import pending_record


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def source(self, sid):
        image = self.root / (sid + '.png')
        cv2.imwrite(str(image), np.full((256, 256), 255, np.uint8))
        return dict(sample_id=sid, character='一', image_path=str(image), source_image_path=str(image))

    def test_bounded_preview_and_relative_path_stream(self):
        consumed = []
        def rows():
            for i in range(10000):
                consumed.append(i)
                yield dict(sample_id=str(i))
        preview(self.root, rows(), 'Preview', limit=2)
        self.assertEqual(consumed, [0, 1])
        self.assertEqual((self.root / 'index.html').read_text().count('<article>'), 2)
        write_records(self.root / 'manifest.jsonl', [dict(image_path='a.png')])
        self.assertEqual(next(iter_records(self.root / 'manifest.jsonl'))['image_path'], str(self.root / 'a.png'))
        self.assertEqual([len(b) for b in chunks(range(7), 3)], [3, 3, 1])

    def test_reads_only_cleanup_header(self):
        report = self.root / 'report.json'
        report.write_text('{\n"processed": 5,\n"samples": [\nNOT PARSED')
        self.assertEqual(cleanup_summary(report)['processed'], 5)

    def test_exclusive_lock(self):
        with exclusive_run(self.root):
            with self.assertRaises(ValueError):
                with exclusive_run(self.root):
                    pass

    def test_stage_error_accounting_reuse_and_tampering(self):
        worker = self.root / 'worker.py'
        worker.write_text('''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args()
out=Path(a.output)
(out/'manifest.jsonl').write_text('{"candidate_id":"a"}\\n')
(out/'errors.jsonl').write_text('{"candidate_id":"b","error":"fixture"}\\n')
raise SystemExit(1)
''')
        output = stage(self.root, 'candidate', sys.executable, str(worker), [],
                       ['manifest.jsonl', 'errors.jsonl'], partial_count=2)
        # A completed stage must be reusable without launching its worker.
        worker.unlink()
        self.assertEqual(stage(self.root, 'candidate', sys.executable, str(worker), [],
                               ['manifest.jsonl', 'errors.jsonl'], partial_count=2), output)
        (output / 'manifest.jsonl').write_text('')
        with self.assertRaises(ValueError):
            checked_output(self.root / 'candidate.done.json')
        worker.write_text('''import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args()
for name in ('manifest.jsonl','errors.jsonl'): (Path(a.output)/name).write_text('')
raise SystemExit(1)
''')
        with self.assertRaises(ValueError):
            stage(self.root, 'incomplete', sys.executable, str(worker), [],
                  ['manifest.jsonl', 'errors.jsonl'], partial_count=2)
        self.assertFalse((self.root / 'incomplete.done.json').exists())

    def test_real_export_of_passing_mask(self):
        row = self.source('passing')
        npz = self.root / 'mask.npz'
        mask = np.zeros((1, 256, 256), np.uint8); mask[:, 50:60, 20:200] = 1
        np.savez_compressed(npz, extractnet_strokes=mask, stroke_labels=np.array([1], np.int64))
        row.update(extracted_path=str(npz), cleanup_status='clean_candidate')
        manifest = self.root / 'cleanup.jsonl'; report = self.root / 'report.json'
        write_records(manifest, [row])
        write_json(report, dict(samples=[dict(sample_id='passing', cleanup_status='clean_candidate', after=dict(iou=.95))]))
        output = stage(self.root, 'export', sys.executable, 'export_clean_masks_for_vs.py',
                       ['--manifest', str(manifest), '--cleanup-report', str(report), '--iou', '.87'],
                       ['manifest.jsonl', 'rejected.jsonl', 'summary.json'])
        self.assertEqual(read_json(output / 'summary.json')['selected_strokes'], 1)
        exported = list(iter_records(output / 'manifest.jsonl'))[0]
        np.testing.assert_array_equal(cv2.imread(exported['input_png'], 0) < 128, mask[0].astype(bool))

    def test_rejected_batch_skips_model_and_merges_all_sources(self):
        source = self.source('below')
        report = self.root / 'sample_cleanup.json'
        write_json(report, dict(sample_id='below', cleanup_status='below_whole_iou', after=dict(iou=.4)))
        row = dict(source, cleanup_report_path=str(report), cleanup_status='below_whole_iou')
        config = dict(pipeline_python=sys.executable, trajectory_style='test')
        batch_dir = self.root / 'batch'
        manifest = run_batch(batch_dir, [row], config, .87, self.root / 'unused_refs', False)
        # Same checkpoint can be read again, without needing any model configuration.
        self.assertEqual(run_batch(batch_dir, [row], config, .87, self.root/'unused_refs', False), manifest)
        ready_source = self.source('ready'); missing_source = self.source('missing')
        ready = pending_record(ready_source, .87, 'test')
        ready['build_status'] = 'trajectory_ready'; ready['quality']['passed'] = True
        ready_manifest = self.root / 'ready.jsonl'; write_records(ready_manifest, [ready])
        sources = self.root / 'sources.jsonl'; write_records(sources, [ready_source, source, missing_source])
        output = self.root / 'dataset'
        merge_datasets([manifest, ready_manifest], sources, output, .87, 'test', True)
        summary = read_json(output / 'summary.json')
        self.assertEqual((summary['samples'], summary['ready'], summary['pending']), (3, 1, 2))
        merged = list(iter_records(output / 'manifest.jsonl'))
        self.assertEqual([r['sample_id'] for r in merged], ['ready', 'below', 'missing'])
        self.assertEqual(summary['quality_passed'], 1)
        self.assertTrue(Path(merged[2]['image_path']).is_file())
        self.assertNotEqual(merged[2]['image_path'], missing_source['image_path'])
        self.assertFalse((output / 'merge.sqlite').exists())


if __name__ == '__main__':
    unittest.main()
