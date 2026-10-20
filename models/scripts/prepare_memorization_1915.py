"""Freeze the original strict 1915 IDs into an independent LMDB for memorization."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import lmdb

ROOT=Path(__file__).resolve().parents[3]
MANIFEST=ROOT/'CHTR/reproduction/pipeline_runs/calligraphy_full_gpu0_20261013_134511_2134141/skeleton_continuation_v1/dataset/training_manifest.jsonl'
SOURCE=ROOT/'CHTR/data/chtr_mmah_lmdb'
DESTINATION=ROOT/'CHTR/data/chtr_mmah_lmdb_1915'


def main():
    rows=[json.loads(line) for line in MANIFEST.open() if line.strip()]
    ids=[r['sample_id'] for r in rows]
    if len(ids)!=1915 or len(set(ids))!=1915:raise ValueError('Expected exactly the original 1915 IDs')
    signature=hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    marker=DESTINATION/'complete.json'
    if marker.exists():
        if json.loads(marker.read_text()).get('source_manifest_sha256')!=signature:raise ValueError('Frozen data signature changed')
        print('Frozen dataset already complete: '+str(DESTINATION));return
    if DESTINATION.exists():raise ValueError('Incomplete destination exists')
    with (SOURCE.parent/'.chtr_mmah_lmdb.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH)
        work=Path(tempfile.mkdtemp(prefix='.chtr_1915_',dir=str(SOURCE.parent)))
        with lmdb.open(str(SOURCE),readonly=True,lock=False,readahead=False) as source:
            with source.begin() as read:
                with lmdb.open(str(work),map_size=1024**3) as dest:
                    with dest.begin(write=True) as write:
                        for sid in ids:
                            value=read.get(sid.encode())
                            if value is None:raise ValueError('Missing original sample: '+sid)
                            row=json.loads(value)
                            for key in ('image_key','trajectory_key','masks_key','original_image_key'):
                                asset=row.get(key)
                                if asset:
                                    data=read.get(asset.encode())
                                    if data is None:raise ValueError('Missing asset: '+asset)
                                    write.put(asset.encode(),data)
                            if not row.get('trajectory_key'):raise ValueError('Missing trajectory: '+sid)
                            write.put(sid.encode(),value)
                        write.put(b'__ids__',json.dumps(ids).encode());write.put(b'__ids_ready__',json.dumps(ids).encode())
                        meta=dict(schema='chtr-lmdb-v1',count=1915,ready=1915,pending=0,include_masks=True,include_original=True,
                                  source_manifest=str(MANIFEST),source_manifest_sha256=signature,frozen=True)
                        write.put(b'__meta__',json.dumps(meta).encode())
                    dest.sync()
        shutil.copyfile(SOURCE/'stroke_type_map.json',work/'stroke_type_map.json')
        (work/'summary.json').write_text(json.dumps(dict(meta,samples=1915),indent=2)+'\n')
        (work/'complete.json').write_text(json.dumps(dict(source_manifest_sha256=signature,samples=1915),indent=2)+'\n')
        work.rename(DESTINATION)
    print('Frozen 1915 samples -> '+str(DESTINATION))


if __name__=='__main__':main()
