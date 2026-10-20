"""Assemble exploratory single-path stroke trajectories into character NPZs."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (arrays, digest, gray, iou, output_dir, png, preview,
                    records, render, require, states, validate_points,
                    write_json, write_records)


def assemble_group(rows, output, threshold):
    rows = sorted(rows, key=lambda row: row['stroke_index'])
    first = rows[0]
    expected = int(first['expected_strokes'])
    require([row['stroke_index'] for row in rows] == list(range(expected)),
            'missing or duplicate stroke trajectories')
    xyz, subpaths, stroke_offsets, labels, per_stroke = [], [0], [0], [], []
    for row in rows:
        z = arrays(row['trajectory_path'])
        validate_points(z['points'], z['subpath_offsets'], bounds=False)
        base = stroke_offsets[-1]
        xyz.append(z['points'][:, :3])
        subpaths.extend((z['subpath_offsets'][1:] + base).tolist())
        stroke_offsets.append(base + len(z['points']))
        labels.append(int(row['stroke_label']))
        per_stroke.append(dict(stroke_index=row['stroke_index'], stroke_id=row['stroke_id'],
                               iou=row['qc']['iou'], single_path_status=row['single_path_status']))
    offsets = np.asarray(subpaths, dtype=np.int64)
    points = states(np.c_[np.concatenate(xyz).astype(np.float32),
                          np.zeros((stroke_offsets[-1], 3), np.float32)], offsets)
    validate_points(points, offsets, bounds=False)
    image = render(points, offsets)
    target = gray(first['image_path']) < 128
    whole = iou(image >= 128, target)
    directory = output / first['sample_id']
    directory.mkdir()
    npz = directory / 'sample.npz'
    np.savez_compressed(npz, points=points, subpath_offsets=offsets,
                        stroke_offsets=np.asarray(stroke_offsets, np.int64),
                        stroke_types=np.asarray(labels, np.int64), image_size=np.array(256),
                        schema=np.array('chtr-single-path-skeleton-v1'))
    render_path = png(directory / 'render.png', 255 - image)
    overlay = np.full((256, 256, 3), 255, np.uint8)
    overlay[target] = (0, 0, 255)
    overlay[image >= 128] = (255, 0, 0)
    overlay[target & (image >= 128)] = (0, 0, 0)
    overlay_path = png(directory / 'overlay.png', overlay)
    item = dict(schema='chtr-single-path-skeleton-v1', route='skeleton_distance_transform',
                sample_id=first['sample_id'], character=first['character'],
                stroke_count=expected, npz_path=str(npz.resolve()), npz_sha256=digest(npz),
                image_path=first['image_path'], render_path=render_path, overlay_path=overlay_path,
                whole_iou=whole, threshold=threshold,
                stroke_qc=per_stroke, single_subpath=True, exploratory=True,
                review_status='auto_selected_exploratory', human_review_performed=False)
    write_json(directory / 'metadata.json', item)
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iou-threshold', type=float, default=.87)
    args = parser.parse_args()
    require(0 < args.iou_threshold <= 1, 'invalid IoU threshold')
    groups = defaultdict(list)
    for row in records(args.manifest):
        groups[row['sample_id']].append(row)
    out = output_dir(args.output)
    packed, rejected = [], []
    for sample_id, rows in sorted(groups.items()):
        try:
            packed.append(assemble_group(rows, out, args.iou_threshold))
        except Exception as exc:
            rejected.append(dict(sample_id=sample_id, error=str(exc)))
    write_records(out / 'manifest.jsonl', packed)
    write_records(out / 'rejected.jsonl', rejected)
    write_json(out / 'summary.json', dict(schema='chtr-single-path-skeleton-v1', samples=len(packed),
        rejected=len(rejected), whole_iou_threshold=args.iou_threshold,
        passed=sum(row['whole_iou'] >= args.iou_threshold for row in packed), exploratory=True))
    preview(out, packed, 'Single-path skeleton character reconstruction')
    print('Packed %d characters; %d rejected' % (len(packed), len(rejected)))
    if rejected:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
