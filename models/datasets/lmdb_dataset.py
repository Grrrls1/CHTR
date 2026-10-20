"""Strict LMDB adapter: one stroke equals one continuous path."""
import io
import json
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


def load_type_map(path):
    mapping = json.loads(Path(path).read_text(encoding='utf-8'))
    types = mapping['types']
    names = {int(t['id']): t['name_zh'] for t in types}
    if len(types) != 24 or set(names) != set(range(24)):
        raise ValueError('stroke_type_map.json must contain exactly IDs 0..23')
    return mapping, names


def normalize_points(points, size):
    result = np.array(points, dtype=np.float32, copy=True)
    result[:, :2] /= size - 1
    result[:, 2] /= size
    return result


def denormalize_points(points, size):
    result = np.array(points, dtype=np.float32, copy=True)
    result[:, :2] *= size - 1
    result[:, 2] *= size
    return result


def validate_sample(points, offsets, subpaths, types, masks, row, names, size):
    def check(condition, message):
        if not condition:
            raise ValueError(row['sample_id'] + ': ' + message)
    k, t = len(types), len(points)
    check(k > 0 and points.shape == (t, 6) and t > 0, 'invalid points/types shape')
    check(np.isfinite(points).all() and (points[:, 2] > 0).all(), 'invalid coordinates or widths')
    check(((points[:, :2] >= 0) & (points[:, :2] <= size-1)).all(), 'out of bounds coordinates')
    check(offsets.shape == (k+1,) and offsets.dtype.kind in 'iu', 'invalid offsets shape/dtype')
    check(offsets[0] == 0 and offsets[-1] == t and (np.diff(offsets) > 0).all(), 'invalid offsets ranges')
    check(np.array_equal(offsets, subpaths), 'one stroke must equal one subpath')
    expected = np.zeros((t, 3), np.float32)
    expected[:, 0] = 1
    expected[offsets[1:]-1] = [0, 1, 0]
    expected[-1] = [0, 0, 1]
    check(np.array_equal(points[:, 3:], expected), 'point states disagree with offsets')
    check(masks.shape == (k, size, size), 'invalid stroke masks shape')
    check(row['stroke_count'] == k and len(row['stroke_map']) == k, 'stroke count mismatch')
    check(np.array_equal(types, row['stroke_types']), 'trajectory/metadata types mismatch')
    stroke_map = sorted(row['stroke_map'], key=lambda s: s['stroke_index'])
    for i, stroke in enumerate(stroke_map):
        label = int(types[i])
        check(label in names, 'unknown stroke label')
        check(stroke['stroke_index'] == i and stroke['stroke_label'] == label, 'stroke index/label mismatch')
        check(stroke['point_range'] == [int(offsets[i]), int(offsets[i+1])], 'point range mismatch')
        check(stroke['stroke_type_name'] == names[label] and row['stroke_type_names'][i] == names[label], 'stroke name mismatch')
        check(bool(masks[i].any()), 'empty stroke mask')


class CHTRModelDataset(Dataset):
    def __init__(self, lmdb_dir, augment=False, scale_range=(0.9, 1.0), translate=3.0):
        self.path = str(Path(lmdb_dir).resolve())
        self.type_map, self.names = load_type_map(Path(self.path) / 'stroke_type_map.json')
        self.augment = augment
        self.scale_range = scale_range
        self.translate = translate
        self.env = None
        with lmdb.open(self.path, readonly=True, lock=False, readahead=False) as env:
            with env.begin() as txn:
                self.all_ids = json.loads(txn.get(b'__ids__'))
                self.meta = json.loads(txn.get(b'__meta__'))
                ready = txn.get(b'__ids_ready__')
                if ready is not None:
                    self.ids = json.loads(ready)
                elif self.meta.get('pending', 0):
                    self.ids = [sid for sid in self.all_ids
                                if json.loads(txn.get(sid.encode())).get('trajectory_key')]
                else:
                    self.ids = self.all_ids
        self.pending_count = len(self.all_ids) - len(self.ids)
        if not self.ids:
            raise ValueError('empty dataset')

    def _env(self):
        if self.env is None:
            self.env = lmdb.open(self.path, readonly=True, lock=False, readahead=False)
        return self.env

    def __getstate__(self):
        state = dict(self.__dict__)
        state['env'] = None
        return state

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        with self._env().begin() as txn:
            row = json.loads(txn.get(sid.encode()))
            def asset(key):
                value = txn.get(row[key].encode()) if row.get(key) else None
                if value is None:
                    raise ValueError(sid + ': missing asset ' + key)
                return value
            image = cv2.imdecode(np.frombuffer(asset('image_key'), np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape[0] != image.shape[1]:
                raise ValueError(sid + ': expected square image')
            size = image.shape[0]
            with np.load(io.BytesIO(asset('trajectory_key')), allow_pickle=False) as z:
                points = z['points'].copy()
                offsets = z['stroke_offsets'].copy()
                subpaths = z['subpath_offsets'].copy()
                types = z['stroke_types'].copy()
                if 'image_size' in z and int(z['image_size']) != size:
                    raise ValueError(sid + ': trajectory canvas mismatch')
            with np.load(io.BytesIO(asset('masks_key')), allow_pickle=False) as z:
                decoded = []
                for i in range(len(types)):
                    name = 'stroke%03d.png' % i
                    if name not in z:
                        raise ValueError(sid + ': missing mask ' + name)
                    mask = cv2.imdecode(z[name], cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise ValueError(sid + ': invalid mask ' + name)
                    # Stored PNGs are white-background black-ink stroke images.
                    decoded.append(mask < 128)
                masks = np.stack(decoded).astype(np.float32)
        validate_sample(points, offsets, subpaths, types, masks, row, self.names, size)
        transform = np.eye(3, dtype=np.float32)
        if self.augment:
            points, image, masks, transform = self._augment(points, image, masks, size)
        normalized = normalize_points(points, size)
        return {
            'image': torch.from_numpy(image.astype(np.float32)[None] / 255.0),
            'points': torch.from_numpy(normalized),
            'stroke_masks': torch.from_numpy(masks),
            'stroke_types': torch.from_numpy(types.astype(np.int64)),
            'stroke_offsets': torch.from_numpy(offsets.astype(np.int64)),
            'stroke_quality': torch.tensor([s['stroke_iou'] for s in sorted(row['stroke_map'], key=lambda s: s['stroke_index'])], dtype=torch.float32),
            'sample_id': sid, 'character': row['character'], 'metadata': row,
            'image_size': size, 'transform': torch.from_numpy(transform),
        }

    def _augment(self, points, image, masks, size):
        # Bound translation using trajectory extrema; never clip target coordinates.
        scale = float(torch.empty(()).uniform_(*self.scale_range))
        xy = (points[:, :2] - (size-1)/2) * scale + (size-1)/2
        translations = []
        for axis in range(2):
            lo = max(-self.translate, -float(xy[:, axis].min()))
            hi = min(self.translate, size-1-float(xy[:, axis].max()))
            if lo > hi:
                raise ValueError('augmentation scale does not fit canvas')
            translations.append(float(torch.empty(()).uniform_(lo, hi)))
        affine = np.array([[scale, 0, (1-scale)*(size-1)/2+translations[0]],
                           [0, scale, (1-scale)*(size-1)/2+translations[1]]], np.float32)
        result = points.copy()
        result[:, :2] = points[:, :2] @ affine[:, :2].T + affine[:, 2]
        result[:, 2] *= scale
        image = cv2.warpAffine(image, affine, (size, size), flags=cv2.INTER_LINEAR, borderValue=255)
        masks = np.stack([cv2.warpAffine(m, affine, (size, size), flags=cv2.INTER_NEAREST) for m in masks])
        matrix = np.eye(3, dtype=np.float32)
        matrix[:2] = affine
        return result, image, masks, matrix


def collate_strokes(samples):
    """Pad the stroke axis and each stroke's point axis independently."""
    b = len(samples)
    k = max(len(s['stroke_types']) for s in samples)
    length = max(int(torch.diff(s['stroke_offsets']).max()) for s in samples)
    h, w = samples[0]['image'].shape[-2:]
    points = torch.zeros(b, k, length, 6)
    masks = torch.zeros(b, k, h, w)
    types = torch.zeros(b, k, dtype=torch.long)
    valid_stroke = torch.zeros(b, k, dtype=torch.bool)
    valid_point = torch.zeros(b, k, length, dtype=torch.bool)
    last = torch.zeros(b, k)
    endpoints = torch.zeros(b, k, 4)
    quality = torch.zeros(b, k)
    for i, s in enumerate(samples):
        count = len(s['stroke_types'])
        valid_stroke[i, :count] = True
        masks[i, :count] = s['stroke_masks']
        types[i, :count] = s['stroke_types']
        quality[i, :count] = s['stroke_quality']
        last[i, count-1] = 1
        for j, (start, end) in enumerate(zip(s['stroke_offsets'][:-1], s['stroke_offsets'][1:])):
            trajectory = s['points'][int(start):int(end)]
            n = len(trajectory)
            points[i, j, :n] = trajectory
            valid_point[i, j, :n] = True
            endpoints[i, j] = torch.cat([trajectory[0, :2], trajectory[-1, :2]])
    return {'images': torch.stack([s['image'] for s in samples]), 'points': points,
            'masks': masks, 'types': types, 'valid_stroke': valid_stroke,
            'valid_point': valid_point, 'last': last, 'endpoints': endpoints,
            'quality': quality, 'sample_ids': [s['sample_id'] for s in samples]}
