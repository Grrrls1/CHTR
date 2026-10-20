"""Readable LMDB dataloader for the CHTR dataset."""
import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    class Dataset(object):
        pass


class CHTRLMDBDataset(Dataset):
    def __init__(self, lmdb_dir, min_iou=None, require_quality_pass=False,
                 require_trajectory=True, limit=None):
        try:
            import lmdb
        except ImportError as exc:
            raise ImportError('请在当前 Python 环境安装 lmdb: pip install lmdb') from exc
        self.path = Path(lmdb_dir).resolve()
        self.env = lmdb.open(str(self.path), readonly=True, lock=False, readahead=False,
                             meminit=False, subdir=True)
        with self.env.begin() as txn:
            ids = json.loads(txn.get(b'__ids__').decode('utf-8'))
            self.meta = json.loads(txn.get(b'__meta__').decode('utf-8'))
            rows = [json.loads(txn.get(sample_id.encode('utf-8')).decode('utf-8')) for sample_id in ids]
        selected = []
        for row in rows:
            quality = row.get('quality') or {}
            score = quality.get('score')
            if require_trajectory and not row.get('trajectory_key'):
                continue
            if min_iou is not None and (score is None or float(score) < min_iou):
                continue
            if require_quality_pass and not quality.get('passed', False):
                continue
            selected.append(row)
        self.rows = selected[:limit] if limit else selected

    def __len__(self):
        return len(self.rows)

    def __del__(self):
        env = getattr(self, 'env', None)
        if env is not None:
            env.close()

    def __getitem__(self, index):
        row = self.rows[index]
        with self.env.begin() as txn:
            image_data = txn.get(row['image_key'].encode('utf-8'))
            trajectory_data = txn.get(row['trajectory_key'].encode('utf-8')) if row.get('trajectory_key') else None
            original_data = txn.get(row['original_image_key'].encode('utf-8')) if row.get('original_image_key') else None
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        item = dict(sample_id=row['sample_id'], character=row['character'],
                    style=row.get('style'), trajectory_style=row.get('trajectory_style'),
                    image=image.astype(np.float32)[None, :, :] / 255.0,
                    stage_iou=row.get('stage_iou', {}), quality=row.get('quality', {}),
                    stroke_map=row.get('stroke_map', []), metadata=row,
                    original_image_bytes=original_data)
        if trajectory_data is None:
            item.update(points=None, subpath_offsets=None, stroke_offsets=None, stroke_types=None)
            return item
        with np.load(io.BytesIO(trajectory_data), allow_pickle=False) as z:
            item.update(points=z['points'], subpath_offsets=z['subpath_offsets'],
                        stroke_offsets=z['stroke_offsets'], stroke_types=z['stroke_types'])
        return item


def collate_keep_variable_length(batch):
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmdb', required=True)
    parser.add_argument('--min-iou', type=float, default=None)
    parser.add_argument('--require-quality-pass', action='store_true')
    parser.add_argument('--allow-pending', action='store_true')
    parser.add_argument('--limit', type=int, default=3)
    args = parser.parse_args()
    dataset = CHTRLMDBDataset(args.lmdb, min_iou=args.min_iou,
                               require_quality_pass=args.require_quality_pass,
                               require_trajectory=not args.allow_pending, limit=args.limit)
    print('lmdb:', Path(args.lmdb).resolve())
    print('records:', len(dataset))
    for index in range(len(dataset)):
        item = dataset[index]
        print(index, item['sample_id'], item['character'],
              'image', tuple(item['image'].shape),
              'points', None if item['points'] is None else tuple(item['points'].shape),
              'quality', item['quality'])


if __name__ == '__main__':
    main()
