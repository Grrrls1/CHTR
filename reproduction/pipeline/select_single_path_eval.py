"""Select a deterministic 12-stroke VS diagnostic set and make review assets."""
import argparse
import base64
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import (arrays, digest, gray, output_dir, png, preview, records,
                    render, require, states, write_records)


def _path_metrics(row):
    target = gray(row['input_png']) < 128
    z = arrays(row['trajectory_path'])
    points = z['points']
    offsets = z['subpath_offsets']
    turns = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        xy = points[start:end, :2]
        if len(xy) < 3:
            continue
        vectors = np.diff(xy, axis=0)
        norms = np.linalg.norm(vectors, axis=1)
        valid = norms > 1e-4
        vectors = vectors[valid] / norms[valid, None]
        if len(vectors) >= 2:
            turns.extend(np.arccos(np.clip(np.sum(vectors[:-1] * vectors[1:], axis=1), -1, 1)))
    ys, xs = np.where(target)
    bbox = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    distance = cv2.distanceTransform(target.astype(np.uint8), cv2.DIST_L2, 5)
    return dict(subpaths=int(len(offsets) - 1), target_area=int(target.sum()),
                mask_width_median=float(2 * np.median(distance[target])),
                mask_width_max=float(2 * distance.max()),
                turn_mean=float(np.mean(turns) if turns else 0.0),
                turn_total=float(np.sum(turns) if turns else 0.0),
                turn_count=int(len(turns)), bbox=int(bbox),
                iou=float(row['qc']['iou']))


def _choose(items, key, reverse=False, count=3, used=None):
    used = used or set()
    ordered = sorted(items, key=lambda item: (key(item), item['stroke_id']), reverse=reverse)
    chosen = [item for item in ordered if item['stroke_id'] not in used][:count]
    used.update(item['stroke_id'] for item in chosen)
    return chosen


def _write_svg(points, offsets, path):
    paths = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        p = points[start:end]
        if len(p) < 2:
            continue
        d = 'M ' + ' '.join('%.2f,%.2f' % tuple(xy) for xy in p[:, :2])
        width = float(np.median(p[:, 2]))
        paths.append('<path d="%s" stroke="#111" stroke-width="%.2f" '
                     'stroke-linecap="round" stroke-linejoin="round" fill="none"/>' % (d, width))
    Path(path).write_text('<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" '
                          'viewBox="0 0 256 256">%s</svg>\n' % ''.join(paths), encoding='utf-8')


def _write_gif(points, offsets, path, target):
    frames = []
    total = len(points)
    for end in np.unique(np.linspace(1, total, min(32, total), dtype=int)):
        prefix = points[:end].copy()
        prefix_offsets = np.array([0, end], dtype=np.int64)
        prefix = states(np.c_[prefix[:, :3], np.zeros((end, 3), np.float32)], prefix_offsets)
        image = render(prefix, prefix_offsets)
        canvas = np.full((256, 256, 3), 255, dtype=np.uint8)
        canvas[target] = (225, 225, 225)
        canvas[image >= 128] = (30, 30, 30)
        frames.append(Image.fromarray(canvas, 'RGB'))
    if not frames:
        return
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=70, loop=0)


def _copy_asset(source, destination):
    if source and Path(source).is_file():
        shutil.copy2(source, destination)
        return str(Path(destination).resolve())
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='vs_qc_v3/best.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--per-group', type=int, default=3)
    args = parser.parse_args()
    require(args.per_group > 0, 'per-group must be positive')
    rows = records(args.manifest)
    require(rows, 'empty manifest')
    require(len({row['stroke_id'] for row in rows}) == len(rows), 'duplicate stroke_id')
    items = []
    for row in rows:
        metrics = _path_metrics(row)
        item = dict(row, selection_metrics=metrics)
        items.append(item)
    used = set()
    selected = []
    simple = _choose(items, lambda x: (x['selection_metrics']['subpaths'],
                                       x['selection_metrics']['turn_total']), count=args.per_group, used=used)
    selected.extend((item, 'simple_line_or_slant') for item in simple)
    folds = _choose(items, lambda x: (x['selection_metrics']['turn_total'],
                                      x['selection_metrics']['turn_mean']), reverse=True,
                    count=args.per_group, used=used)
    selected.extend((item, 'fold_or_turn') for item in folds)
    wide = _choose(items, lambda x: (x['selection_metrics']['mask_width_median'],
                                     x['selection_metrics']['target_area']), reverse=True,
                   count=args.per_group, used=used)
    selected.extend((item, 'wide_calligraphy') for item in wide)
    multi = _choose(items, lambda x: (x['selection_metrics']['subpaths'],
                                      -x['selection_metrics']['iou']), reverse=True,
                    count=args.per_group, used=used)
    selected.extend((item, 'multi_subpath_failure') for item in multi)
    require(len(selected) == args.per_group * 4,
            'Could not select enough distinct diagnostic samples')
    out = output_dir(args.output)
    result = []
    for item, category in selected:
        source_dir = out / item['stroke_id']
        source_dir.mkdir()
        z = arrays(item['trajectory_path'])
        copied = dict(input_png=_copy_asset(item['input_png'], source_dir / 'target.png'),
                      upstream_render=_copy_asset(item.get('upstream_render_path'), source_dir / 'upstream_render.png'),
                      render=_copy_asset(item.get('render_path'), source_dir / 'render.png'),
                      overlay=_copy_asset(item.get('overlay_path'), source_dir / 'overlay.png'))
        svg = source_dir / 'trajectory.svg'
        gif = source_dir / 'trajectory.gif'
        _write_svg(z['points'], z['subpath_offsets'], svg)
        _write_gif(z['points'], z['subpath_offsets'], gif, gray(item['input_png']) < 128)
        result.append(dict(item, diagnostic_category=category,
                           source_candidate_sha256=digest(item['npz']),
                           source_trajectory_sha256=digest(item['trajectory_path']),
                           assets=dict(**copied, svg=str(svg.resolve()), gif=str(gif.resolve())),
                           exploratory=True))
    result.sort(key=lambda row: (row['diagnostic_category'], row['stroke_id']))
    write_records(out / 'selection.jsonl', result)
    preview(out, result, 'VS single-path diagnostic selection')
    print('Selected %d diagnostic strokes' % len(result))


if __name__ == '__main__':
    main()
