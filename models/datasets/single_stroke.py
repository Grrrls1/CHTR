"""Stage-two data: fixed character split, original grayscale under each stroke mask."""
import bisect
import io
import json
import os
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_split import validate_split, read_candidates


class StrokeCatalog:
    """Validate once, share metadata and a read-only process-local LMDB handle."""
    def __init__(self, dataset, split_file):
        self.path = str(Path(dataset).resolve())
        self.split_file = str(Path(split_file).resolve())
        self.split = json.loads(Path(split_file).read_text())
        filters = self.split.get('filters', {})
        rows, self.selection = read_candidates(self.path, **filters)
        validate_split(self.split, rows)
        self.rows = {r['sample_id']: r for r in rows}
        self.fingerprint = self.split['dataset_fingerprint']
        self.env = None
        self.env_pid = None
        self.cached_id = None
        self.cached_assets = None

    def _env(self):
        if self.env is not None and self.env_pid != os.getpid():
            self.close()
        if self.env is None:
            self.env = lmdb.open(self.path, readonly=True, lock=False, readahead=False)
            self.env_pid = os.getpid()
        return self.env

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(env=None, env_pid=None, cached_id=None, cached_assets=None)
        return state

    def close(self):
        if self.env is not None:
            self.env.close()
        self.env = None
        self.env_pid = None
        self.cached_id = self.cached_assets = None

    def character_assets(self, sid):
        if self.env_pid is not None and self.env_pid != os.getpid():
            self.close()
        if self.cached_id == sid:
            return self.cached_assets
        row = self.rows[sid]
        with self._env().begin() as txn:
            data = {}
            for key in ('image_key', 'trajectory_key', 'masks_key'):
                value = txn.get(row[key].encode())
                if value is None:
                    raise ValueError(sid+': missing '+key)
                data[key] = value
        image = cv2.imdecode(np.frombuffer(data['image_key'], np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (256, 256):
            raise ValueError(sid+': expected normalized 256x256 image')
        with np.load(io.BytesIO(data['trajectory_key']), allow_pickle=False) as z:
            points = z['points'][:, :3].astype(np.float32, copy=True)
            offsets = z['stroke_offsets'].copy()
            if int(z['image_size']) != 256:
                raise ValueError(sid+': trajectory canvas mismatch')
        n = row['stroke_count']
        if (offsets.shape != (n+1,) or offsets.dtype.kind not in 'iu' or offsets[0] != 0
                or offsets[-1] != len(points) or (np.diff(offsets) < 1).any()):
            raise ValueError(sid+': invalid stroke offsets')
        if (not np.isfinite(points).all() or (points[:, :2] < 0).any()
                or (points[:, :2] > 255).any() or (points[:, 2] <= 0).any()):
            raise ValueError(sid+': invalid XY or width')
        stroke_map = sorted(row['stroke_map'], key=lambda s:s['stroke_index'])
        if len(stroke_map) != n or any(s['stroke_index'] != i or s['point_range'] != offsets[i:i+2].tolist()
                                      for i,s in enumerate(stroke_map)):
            raise ValueError(sid+': metadata and actual trajectory offsets differ')
        masks = []
        with np.load(io.BytesIO(data['masks_key']), allow_pickle=False) as z:
            for i in range(n):
                name = 'stroke%03d.png' % i
                if name not in z:
                    raise ValueError(sid+': missing '+name)
                png = cv2.imdecode(z[name], cv2.IMREAD_GRAYSCALE)
                if png is None or png.shape != image.shape:
                    raise ValueError(sid+': invalid '+name)
                mask = png < 128
                if not mask.any():
                    raise ValueError(sid+': empty '+name)
                masks.append(mask)
        result = image, points, offsets, np.stack(masks)
        self.cached_id, self.cached_assets = sid, result
        return result


def masked_stroke_image(image, mask):
    """Shared GT/predicted-mask input construction; mask is bool or probability."""
    if image.shape != (256, 256) or mask.shape != image.shape:
        raise ValueError('image and mask must share a 256x256 canvas')
    if not np.isfinite(mask).all():
        raise ValueError('nonfinite stroke mask')
    return np.where(mask >= .5, image, 255).astype(np.uint8)


class SingleStrokeDataset(Dataset):
    def __init__(self, catalog, split, max_points=2048, augment=False):
        if split not in ('train', 'val', 'test'):
            raise ValueError('unknown split')
        if augment and split != 'train':
            raise ValueError('random augmentation is only allowed for train')
        self.catalog, self.split, self.max_points, self.augment = catalog, split, max_points, augment
        self.ids = list(catalog.split[split+'_ids'])
        self.ends = np.cumsum([catalog.rows[s]['stroke_count'] for s in self.ids]).tolist()
        if not self.ends or self.ends[-1] == 0:
            raise ValueError('empty stroke dataset')
        for sid in self.ids:
            for s in catalog.rows[sid]['stroke_map']:
                length = s['point_range'][1]-s['point_range'][0]
                if not 1 <= length <= max_points:
                    raise ValueError(f'{sid}: stroke {s["stroke_index"]} has {length} points; cap {max_points}; no truncation')

    def __len__(self):
        return self.ends[-1]

    def key(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        char_index = bisect.bisect_right(self.ends, index)
        start = self.ends[char_index-1] if char_index else 0
        return self.ids[char_index], index-start

    def __getitem__(self, index):
        sid, k = self.key(index)
        image, points, offsets, masks = self.catalog.character_assets(sid)
        points = points[offsets[k]:offsets[k+1]].copy()
        if not 1 <= len(points) <= self.max_points:
            raise ValueError(sid+': actual points exceed cap; no truncation')
        mask = masks[k]
        transform = np.eye(3, dtype=np.float32)
        if self.augment:
            scale = float(torch.empty(()).uniform_(.9, 1.))
            xy = (points[:, :2]-127.5)*scale+127.5
            shift = [float(torch.empty(()).uniform_(max(-3.,-float(xy[:,a].min())),
                                                    min(3.,255.-float(xy[:,a].max())))) for a in range(2)]
            transform[:2,:2] *= scale
            transform[:2,2] = (1-scale)*127.5+np.array(shift)
            image = cv2.warpAffine(image, transform[:2], (256,256), flags=cv2.INTER_LINEAR, borderValue=255)
            mask = cv2.warpAffine(mask.astype(np.uint8), transform[:2], (256,256), flags=cv2.INTER_NEAREST).astype(bool)
            points[:,:2] = points[:,:2] @ transform[:2,:2].T + transform[:2,2]
            points[:,2] *= scale
        stroke = masked_stroke_image(image, mask)
        points[:,:2] /= 255.
        points[:,2] /= 256.
        return dict(image=torch.from_numpy(stroke.astype(np.float32)[None]/255.),
                    points=torch.from_numpy(points), sample_id=sid, stroke_index=k,
                    character=self.catalog.rows[sid]['character'], split=self.split,
                    transform=torch.from_numpy(transform))


def collate_single_strokes(samples):
    """At output position j, input p[j] predicts p[j+1], or terminal invalid."""
    lengths = torch.tensor([len(s['points']) for s in samples], dtype=torch.long)
    length = int(lengths.max()); batch = len(samples)
    history = torch.zeros(batch, length, 3)
    targets = torch.zeros_like(history)
    active = torch.arange(length)[None] < lengths[:,None]
    valid = torch.arange(length)[None] < (lengths-1)[:,None]
    for i,s in enumerate(samples):
        n = len(s['points'])
        history[i,:n] = s['points']
        targets[i,:n-1] = s['points'][1:]
    return dict(images=torch.stack([s['image'] for s in samples]), history=history,
                targets=targets, active=active, valid=valid, lengths=lengths,
                keys=[(s['sample_id'], s['stroke_index']) for s in samples])
