"""Indexed raw POT catalog shared by the two training stages (no raster cache on disk)."""
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import cv2
import numpy as np

from .casia_pot import iter_pot, normalized_strokes

SCHEMA = 'casia_olhwdb11_xyw_v1'
DEFAULT_RENDER = dict(size=256, margin=8, supersample=4, train_width_min=2,
                      train_width_max=12, eval_width=8, seed=42,
                      width_distribution='discrete_uniform_per_character_per_epoch',
                      mask_threshold=0.5, width_units='diameter_in_256_canvas_pixels')


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_render(p):
    if p.keys() != DEFAULT_RENDER.keys():
        raise ValueError('unknown/missing CASIA rendering parameters')
    for key in ('size', 'margin', 'supersample', 'train_width_min', 'train_width_max', 'eval_width', 'seed'):
        if type(p[key]) is not int:
            raise ValueError('render parameter must be integer: '+key)
    if (p['size'] != 256 or p['supersample'] < 1 or p['supersample'] > 8
            or not 1 <= p['train_width_min'] <= p['train_width_max'] <= 2*p['margin']
            or not 1 <= p['eval_width'] <= 2*p['margin'] or not 0 < p['margin'] < 128
            or p['mask_threshold'] != .5 or p['seed'] < 0
            or p['width_distribution'] != DEFAULT_RENDER['width_distribution']
            or p['width_units'] != DEFAULT_RENDER['width_units']):
        raise ValueError('invalid CASIA rendering protocol')


def render_paths(strokes, width, protocol):
    """Round caps, isotropic geometry, area supersampling; image and masks share coverage."""
    size, ss = protocol['size'], protocol['supersample']
    grayscale = []
    for stroke in strokes:
        canvas = np.zeros((size*ss, size*ss), np.uint8)
        xy = np.rint(np.asarray(stroke)[:, :2]*ss).astype(np.int32)
        cv2.polylines(canvas, [xy], False, 255, max(1, round(width*ss)), cv2.LINE_8)
        for point in (xy[0], xy[-1]):
            cv2.circle(canvas, tuple(point), max(1, round(width*ss/2)), 255, -1, cv2.LINE_8)
        grayscale.append(cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA))
    coverage = np.stack(grayscale)
    image = 255-coverage.max(0)
    return image, coverage >= 128


class CatalogRows(Mapping):
    def __init__(self, catalog):
        self.catalog = catalog

    def __len__(self):
        return sum(len(self.catalog.split[s+'_ids']) for s in ('train', 'val', 'test'))

    def __iter__(self):
        for s in ('train', 'val', 'test'):
            yield from self.catalog.split[s+'_ids']

    def __getitem__(self, sid):
        result = self.catalog.connection().execute(
            'SELECT character,writer,source,record_index,byte_offset,lengths,split FROM samples WHERE sid=?',
            (sid,)).fetchone()
        if result is None:
            raise KeyError(sid)
        character, writer, source, index, offset, lengths, split = result
        offsets = np.r_[0, np.cumsum(json.loads(lengths))].tolist()
        return dict(sample_id=sid, character=character, writer_id=writer, source=source,
                    record_index=index, byte_offset=offset, split=split, stroke_count=len(offsets)-1,
                    stroke_map=[dict(stroke_index=i, point_range=[a,b])
                                for i,(a,b) in enumerate(zip(offsets[:-1],offsets[1:]))])


class CasiaCatalog:
    def __init__(self, dataset, split_file, render=None):
        self.path = str(Path(dataset).resolve())
        self.split_file = str(Path(split_file).resolve())
        self.split = json.loads(Path(split_file).read_text())
        if self.split.get('schema') != SCHEMA:
            raise ValueError('not a CASIA training manifest')
        payload = {k:v for k,v in self.split.items() if k not in ('dataset_fingerprint', 'index_file')}
        if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != self.split['dataset_fingerprint']:
            raise ValueError('CASIA manifest fingerprint mismatch')
        self.protocol = dict(self.split['render'])
        if render:
            self.protocol.update(render)
        validate_render(self.protocol)
        self.fingerprint = self.split['dataset_fingerprint']
        self.db_path = Path(self.split_file).parent/self.split['index_file']
        self._connection = self._pid = None
        self._cache_key = self._assets = None
        if sha256_file(self.db_path) != self.split['index_sha256']:
            raise ValueError('CASIA index fingerprint mismatch')
        for source in self.split['sources']:
            p = Path(self.path)/source['path']
            if not p.is_file() or sha256_file(p) != source['sha256']:
                raise ValueError('CASIA source fingerprint mismatch: '+str(p))
        seen, writers = set(), []
        self.stroke_counts, self.characters = {}, {}
        for s in ('train', 'val', 'test'):
            ids = self.split[s+'_ids']
            if not ids or len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError('empty, repeated or overlapping CASIA split')
            seen.update(ids)
            actual = self.connection().execute('SELECT sid,writer,character,json_array_length(lengths) FROM samples WHERE split=? ORDER BY sid', (s,)).fetchall()
            if [r[0] for r in actual] != ids:
                raise ValueError('CASIA index/manifest split mismatch')
            writers.append({r[1] for r in actual})
            self.stroke_counts.update((r[0],r[3]) for r in actual)
            self.characters.update((r[0],r[2]) for r in actual)
        if any(writers[a] & writers[b] for a,b in ((0,1),(0,2),(1,2))):
            raise ValueError('writer leakage in CASIA splits')
        self.rows = CatalogRows(self)

    def validate_point_cap(self, split, max_points):
        maximum = self.connection().execute(
            'SELECT max(CAST(j.value AS INTEGER)) FROM samples,json_each(samples.lengths) j WHERE split=?',
            (split,)).fetchone()[0]
        if maximum is None or maximum > max_points:
            raise ValueError(f'{split}: stroke has {maximum} points; cap {max_points}; no truncation')

    def connection(self):
        if self._pid != os.getpid():
            self.close()
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path.as_uri()+'?mode=ro&immutable=1', uri=True)
            self._pid = os.getpid()
        return self._connection

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_connection=None, _pid=None, _cache_key=None, _assets=None)
        return state

    def close(self):
        if self._connection is not None:
            self._connection.close()
        self._connection = self._pid = self._cache_key = self._assets = None

    def width_for(self, sid, epoch):
        if epoch is None:
            return self.protocol['eval_width']
        if type(epoch) is not int or epoch < 0:
            raise ValueError('render epoch must be nonnegative')
        token = f'{self.protocol["seed"]}:{epoch}:{sid}'.encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], 'little'))
        return int(rng.integers(self.protocol['train_width_min'], self.protocol['train_width_max']+1))

    def render_assets(self, sid, split, epoch=None):
        row = self.rows[sid]
        if split != row['split'] or (epoch is not None and split != 'train'):
            raise ValueError('random rendering is only allowed for training samples')
        width = self.width_for(sid, epoch)
        key = (sid, width)
        if self._pid == os.getpid() and self._cache_key == key:
            return self._assets
        raw = next(iter_pot(Path(self.path)/row['source'], start_offset=row['byte_offset'],
                            start_index=row['record_index']))
        if raw['index'] != row['record_index'] or raw['char'] != row['character']:
            raise ValueError('POT/index sample mismatch')
        strokes, _ = normalized_strokes(raw['strokes'], self.protocol['size'], self.protocol['margin'])
        lengths = [len(s) for s in strokes]
        if lengths != [s['point_range'][1]-s['point_range'][0] for s in row['stroke_map']]:
            raise ValueError('POT/index trajectory length mismatch')
        image, masks = render_paths(strokes, width, self.protocol)
        xy = np.concatenate(strokes)
        points = np.column_stack((xy, np.full(len(xy), width, np.float32))).astype(np.float32)
        offsets = np.r_[0, np.cumsum(lengths)]
        self._cache_key, self._assets = key, (image, points, offsets, masks)
        return self._assets

    def character_assets(self, sid):
        # Diagnostic/visualization reads always use the frozen evaluation rendering.
        return self.render_assets(sid, self.rows[sid]['split'])
