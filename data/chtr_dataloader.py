"""Readable dataloader for the simplified CHTR manifest.

The manifest keeps every source record.  Training-time selection is controlled
by ``min_iou`` and ``require_quality_pass`` here, not by data construction.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    class Dataset(object):
        pass


class CHTRDataset(Dataset):
    def __init__(self, manifest, min_iou=None, require_quality_pass=False,
                 require_trajectory=True, limit=None):
        self.manifest_path = Path(manifest).resolve()
        self.root = self.manifest_path.parent
        rows = [json.loads(line) for line in self.manifest_path.read_text(encoding='utf-8').splitlines()
                if line.strip()]
        selected = []
        for row in rows:
            quality = row.get('quality') or {}
            score = quality.get('score')
            if require_trajectory and row.get('build_status', 'trajectory_ready') != 'trajectory_ready':
                continue
            if min_iou is not None and (score is None or float(score) < min_iou):
                continue
            if require_quality_pass and not quality.get('passed', False):
                continue
            selected.append(row)
        self.rows = selected[:limit] if limit else selected

    def __len__(self):
        return len(self.rows)

    def _path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self._path(row['image_path'])
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        image = image.astype(np.float32) / 255.0
        if not row.get('trajectory_path'):
            return dict(sample_id=row['sample_id'], character=row['character'],
                        style=row.get('style', 'trajectory_style'),
                        trajectory_style=row.get('trajectory_style'), image=image[None, :, :],
                        points=None, subpath_offsets=None, stroke_offsets=None,
                        stroke_types=None, stroke_type_names=row.get('stroke_type_names'),
                        stage_iou=row.get('stage_iou', {}), quality=row.get('quality', {}),
                        stroke_map=row.get('stroke_map', []), metadata=row)
        trajectory_path = self._path(row['trajectory_path'])
        with np.load(str(trajectory_path), allow_pickle=False) as z:
            trajectory = {key: z[key] for key in z.files}
        return dict(sample_id=row['sample_id'], character=row['character'],
                    style=row.get('style', 'trajectory_style'),
                    trajectory_style=row.get('trajectory_style'), image=image[None, :, :],
                    points=trajectory['points'], subpath_offsets=trajectory['subpath_offsets'],
                    stroke_offsets=trajectory['stroke_offsets'], stroke_types=trajectory['stroke_types'],
                    stroke_type_names=row.get('stroke_type_names'), stage_iou=row.get('stage_iou', {}),
                    quality=row.get('quality', {}), stroke_map=row.get('stroke_map', []), metadata=row)


def collate_keep_variable_length(batch):
    """Keep variable-length trajectories as a list; stack only in model code."""
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--min-iou', type=float, default=None)
    parser.add_argument('--require-quality-pass', action='store_true')
    parser.add_argument('--allow-pending', action='store_true')
    parser.add_argument('--limit', type=int, default=3)
    args = parser.parse_args()
    dataset = CHTRDataset(args.manifest, min_iou=args.min_iou,
                           require_quality_pass=args.require_quality_pass,
                           require_trajectory=not args.allow_pending, limit=args.limit)
    print('manifest:', Path(args.manifest).resolve())
    print('records:', len(dataset))
    for index in range(len(dataset)):
        item = dataset[index]
        print(index, item['sample_id'], item['character'], 'image', tuple(item['image'].shape),
              'points', tuple(item['points'].shape), 'strokes', len(item['stroke_offsets']) - 1,
              'quality', item['quality'])


if __name__ == '__main__':
    main()
