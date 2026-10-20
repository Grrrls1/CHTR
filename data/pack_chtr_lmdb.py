"""Pack a simplified CHTR manifest and its assets into LMDB."""
import argparse
import io
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduction' / 'pipeline'))
from common import arrays, digest, output_dir, records, require, write_json


def file_bytes(path):
    return Path(path).read_bytes() if path and Path(path).is_file() else None


def npz_bytes(path):
    return Path(path).read_bytes() if path and Path(path).is_file() else None


def pack_masks(row, manifest_root):
    masks = {}
    for item in row.get('stroke_map', []):
        value = item.get('mask_path')
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = manifest_root / path
        data = file_bytes(path)
        if data is not None:
            masks['stroke%03d.png' % int(item['stroke_index'])] = np.frombuffer(data, dtype=np.uint8)
    if not masks:
        return None
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **masks)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True, help='New LMDB directory')
    parser.add_argument('--map-size-gb', type=float, default=64)
    parser.add_argument('--progress-every', type=int, default=1000)
    parser.add_argument('--no-original', action='store_true')
    parser.add_argument('--no-masks', action='store_true')
    args = parser.parse_args()
    require(args.map_size_gb > 0 and args.progress_every > 0, 'invalid LMDB options')
    try:
        import lmdb
    except ImportError as exc:
        raise SystemExit('Missing lmdb; install it in the selected Python environment: pip install lmdb') from exc
    rows = records(args.manifest)
    require(rows, 'empty manifest')
    out = output_dir(args.output)
    env = lmdb.open(str(out), map_size=int(args.map_size_gb * 1024 ** 3), subdir=True,
                    map_async=True, writemap=False, lock=True)
    manifest_root = Path(args.manifest).resolve().parent
    ids = []
    ready = pending = 0
    with env.begin(write=True) as txn:
        for index, row in enumerate(rows, 1):
            sample_id = row['sample_id']
            key = sample_id.encode('utf-8')
            require(sample_id not in ids, 'duplicate sample_id: ' + sample_id)
            meta = dict(row, lmdb_key=sample_id, image_key=sample_id + '/image',
                        original_image_key=(None if args.no_original else sample_id + '/original'),
                        trajectory_key=(sample_id + '/trajectory' if row.get('trajectory_path') else None),
                        masks_key=(sample_id + '/masks' if not args.no_masks else None))
            image_path = row.get('image_path')
            if image_path:
                image_data = file_bytes(image_path)
                if image_data is not None:
                    txn.put((sample_id + '/image').encode(), image_data)
            if not args.no_original and row.get('original_image_path'):
                original_data = file_bytes(row['original_image_path'])
                if original_data is not None:
                    txn.put((sample_id + '/original').encode(), original_data)
            if row.get('trajectory_path'):
                trajectory_data = npz_bytes(row['trajectory_path'])
                if trajectory_data is not None:
                    txn.put((sample_id + '/trajectory').encode(), trajectory_data)
                    ready += 1
                else:
                    pending += 1
            else:
                pending += 1
            if not args.no_masks:
                mask_data = pack_masks(row, manifest_root)
                if mask_data is not None:
                    txn.put((sample_id + '/masks').encode(), mask_data)
            txn.put(key, json.dumps(meta, ensure_ascii=False, allow_nan=False).encode('utf-8'))
            ids.append(sample_id)
            if index == 1 or index % args.progress_every == 0 or index == len(rows):
                print('[lmdb] %d/%d ready=%d pending=%d' % (index, len(rows), ready, pending), flush=True)
        txn.put(b'__ids__', json.dumps(ids, ensure_ascii=False).encode('utf-8'))
        txn.put(b'__meta__', json.dumps(dict(schema='chtr-lmdb-v1', manifest=str(Path(args.manifest).resolve()),
                                              count=len(ids), ready=ready, pending=pending,
                                              include_original=not args.no_original,
                                              include_masks=not args.no_masks), ensure_ascii=False).encode('utf-8'))
    env.sync(); env.close()
    write_json(out / 'summary.json', dict(schema='chtr-lmdb-v1', samples=len(ids), ready=ready,
        pending=pending, map_size_gb=args.map_size_gb, include_original=not args.no_original,
        include_masks=not args.no_masks, source_manifest=str(Path(args.manifest).resolve())))
    print('[lmdb] wrote %d samples to %s' % (len(ids), out), flush=True)


if __name__ == '__main__':
    main()
