import fcntl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

PIPE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PIPE));sys.path.insert(0,str(PIPE.parents[1]/'data'))
from common import digest,iter_records,states,write_json,write_records,read_json
from repack_all_clean import rebuild_batch,merge_batches,pack_all,publish
from continue_skeleton_dataset import TYPE_MAP
from models.datasets.lmdb_dataset import CHTRModelDataset,load_type_map


class AllCleanRepackTests(unittest.TestCase):
    def test_low_iou_and_missing_trajectory_are_both_in_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch=root/'oldbatch';batch.mkdir()
            export=batch/'export';export.mkdir();raw=batch/'skeleton';raw.mkdir()
            image=root/'image.png';mask=np.full((256,256),255,np.uint8)
            cv2.line(mask,(35,128),(220,128),0,9);cv2.imwrite(str(image),mask)
            sources=[];exports=[];reports=[]
            for sid in ('low_iou','pending'):
                source=dict(sample_id=sid,character='一',image_path=str(image),source_image_path=str(image))
                sources.append(source)
                exports.append(dict(source,stroke_id=sid+'_stroke000',stroke_index=0,stroke_label=1,
                                    input_png=str(image),expected_strokes=1))
                reports.append(dict(sample_id=sid,before=dict(iou=.95),after=dict(iou=.95)))
            write_records(batch/'cleanup.jsonl',sources);write_json(batch/'cleanup_report.json',dict(samples=reports))
            write_records(export/'manifest.jsonl',exports)
            write_json(batch/'export.done.json',dict(output=str(export),hashes={'manifest.jsonl':digest(export/'manifest.jsonl')}))
            points=states(np.array([[35,128,10,0,0,0],[220,128,10,0,0,0]],np.float32),np.array([0,2],np.int64))
            trajectory=raw/'trajectory.npz';np.savez_compressed(trajectory,points=points,subpath_offsets=np.array([0,2],np.int64))
            # Explicit low quality must not prevent otherwise valid geometry being assembled.
            stroke=dict(exports[0],trajectory_path=str(trajectory),single_path_status='ambiguous',qc=dict(iou=.2))
            write_records(raw/'manifest.jsonl',[stroke])
            write_records(raw/'errors.jsonl',[dict(stroke_id='pending_stroke000',error='Skeleton has fewer than two pixels')])
            write_json(batch/'skeleton.done.json',dict(output=str(raw),hashes={name:digest(raw/name) for name in ('manifest.jsonl','errors.jsonl')}))
            _,names=load_type_map(TYPE_MAP)
            config=dict(quality_threshold=.87,trajectory_style='test')
            with ThreadPoolExecutor(max_workers=1) as pool:
                manifest=rebuild_batch(batch,root/'newbatch',names,config,pool)
                self.assertEqual(rebuild_batch(batch,root/'newbatch',names,config,pool),manifest)
            rows=list(iter_records(manifest));self.assertEqual(len(rows),2)
            self.assertTrue(rows[0]['training_eligible'])
            self.assertFalse(rows[0]['quality']['all_strokes_passed'])
            self.assertEqual(rows[1]['build_status'],'trajectory_pending')
            self.assertEqual(len(rows[1]['stroke_map']),1)
            summary=merge_batches([manifest],root/'merged',2)
            self.assertEqual((summary['samples'],summary['ready'],summary['pending']),(2,1,1))
            lmdb=root/'all_clean_lmdb';pack_all(root/'merged/manifest.jsonl',lmdb,TYPE_MAP,summary,map_gb=.05)
            self.assertEqual(read_json(lmdb/'summary.json')['samples'],2)
            ds=CHTRModelDataset(lmdb)
            self.assertEqual(len(ds.all_ids),2);self.assertEqual(len(ds),1);self.assertEqual(ds.pending_count,1)
            sample=ds[0];self.assertEqual(sample['sample_id'],'low_iou')
            self.assertLess(float(sample['stroke_quality'][0]),.87)
            ds.close()

    def test_publish_keeps_backup_and_respects_training_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'chtr_mmah_lmdb';old.mkdir();(old/'data').write_text('old')
            new=root/'build';new.mkdir();(new/'data').write_text('new')
            with (root/'.chtr_mmah_lmdb.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_SH)
                with self.assertRaises(BlockingIOError):publish(new,old)
                self.assertEqual((old/'data').read_text(),'old')
            publish(new,old)
            self.assertEqual((old/'data').read_text(),'new')
            backups=list(root.glob('chtr_mmah_lmdb.backup_*'));self.assertEqual(len(backups),1)
            self.assertEqual((backups[0]/'data').read_text(),'old')


if __name__=='__main__':unittest.main()
