"""Pack RHSEDB image/6D trajectory pairs after explicit 0.7 experimental QC."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil

import numpy as np

from common import SCHEMA, arrays, digest, gray, iou, output_dir, png, records, render, require, states, validate_points, write_json, write_records


def pack_group(rows, out, minimum=0.7):
    rows = sorted(rows, key=lambda r: r['stroke_index'])
    first = rows[0]
    count = first['expected_strokes']
    require([r['stroke_index'] for r in rows] == list(range(count)), 'missing or duplicate strokes')
    require(first['source'] == 'RHSEDB' and first['category_system'] == 'StrokeExtraction-24', 'unexpected source/category')
    points, subpaths, stroke_offsets, labels, stroke_scores = [], [0], [0], [], []
    for r in rows:
        for key in ('sample_id', 'character', 'image_path', 'source_npz_path', 'source_npz_sha256', 'expected_strokes'):
            require(r[key] == first[key], 'inconsistent ' + key)
        require(r['qc']['qc_status'] == 'passed', 'stroke QC rejected')
        require(r['qc']['iou'] >= minimum and r['qc']['upstream_iou'] >= minimum, 'stroke IoU below %.2f' % minimum)
        require(r['qc'].get('native_replay_iou', 0) >= .999, 'native replay mismatch')
        require(r['qc']['threshold'] == minimum and r['qc']['fidelity_threshold'] == minimum, 'wrong experimental threshold')
        require(digest(r['trajectory_path']) == r['trajectory_sha256'], 'trajectory hash changed')
        require(digest(r['input_png']) == r['input_sha256'], 'mask hash changed')
        require(digest(r['upstream_render_path']) == r['upstream_render_sha256'], 'upstream render hash changed')
        z = arrays(r['trajectory_path'])
        validate_points(z['points'], z['subpath_offsets'])
        target = gray(r['input_png']) < 128
        score = iou(render(z['points'], z['subpath_offsets']) >= 128, target)
        require(score >= minimum, 'recomputed stroke IoU below threshold')
        fidelity = iou(render(z['points'], z['subpath_offsets']) >= 128, gray(r['upstream_render_path']) < 128)
        require(fidelity >= minimum, 'recomputed upstream fidelity below threshold')
        points.append(z['points'])
        subpaths.extend((z['subpath_offsets'][1:] + stroke_offsets[-1]).tolist())
        stroke_offsets.append(stroke_offsets[-1] + len(z['points']))
        labels.append(r['stroke_label'])
        stroke_scores.append(dict(index=r['stroke_index'], target_iou=score,
                                  upstream_iou=r['qc']['upstream_iou'],
                                  native_replay_iou=r['qc']['native_replay_iou']))
    require(digest(first['source_npz_path']) == first['source_npz_sha256'], 'source RHSEDB changed')
    offsets = np.asarray(subpaths, np.int64)
    all_points = states(np.concatenate(points).astype(np.float32), offsets)
    whole = render(all_points, offsets)
    target_image = gray(first['image_path']) < 128
    whole_iou = iou(whole >= 128, target_image)
    require(whole_iou >= minimum, 'whole-character IoU below %.2f: %.4f' % (minimum, whole_iou))
    dest = out / first['sample_id']
    dest.mkdir()
    npz = dest / 'sample.npz'
    np.savez_compressed(npz, points=all_points, subpath_offsets=offsets,
                        stroke_offsets=np.asarray(stroke_offsets, np.int64),
                        stroke_types=np.asarray(labels, np.int64), image_size=np.array(256),
                        schema=np.array('rhsedb-6d-exploratory-v1'))
    image_path = dest / 'image.png'
    shutil.copyfile(first['image_path'], image_path)
    return dict(schema='rhsedb-6d-exploratory-v1', sample_id=first['sample_id'],
                character=first['character'], image_path=str(image_path.resolve()),
                image_sha256=digest(image_path), npz_path=str(npz.resolve()),
                npz_sha256=digest(npz), source_npz_path=first['source_npz_path'],
                source_npz_sha256=first['source_npz_sha256'], source_split=first['source_split'],
                writer_status='unknown_in_local_npz', dataset_split='unassigned',
                qc_threshold=minimum, whole_iou=whole_iou, stroke_count=count,
                stroke_scores=stroke_scores, category_system='StrokeExtraction-24',
                annotation_source='RHSEDB target_single_image',
                trajectory_source='Virtual Sketching reconstructed',
                manual_review_status='not_performed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, help='QC best.jsonl')
    p.add_argument('--output', required=True)
    p.add_argument('--iou', type=float, default=.7)
    args = p.parse_args()
    require(0 < args.iou <= 1, 'invalid IoU')
    groups = defaultdict(list)
    for row in records(args.manifest):
        groups[row['sample_id']].append(row)
    out = output_dir(args.output)
    packed, rejected = [], []
    for sid, rows in groups.items():
        try:
            packed.append(pack_group(rows, out, args.iou))
        except (ValueError, KeyError, OSError) as exc:
            rejected.append(dict(sample_id=sid, error=str(exc)))
    write_records(out / 'manifest.jsonl', packed)
    write_records(out / 'rejected.jsonl', rejected)
    write_json(out / 'summary.json', dict(input_images=len(groups), packed=len(packed),
               rejected=len(rejected), qc_threshold=args.iou, dataset_split='unassigned',
               manual_review='not_performed',
               note='Exploratory RHSEDB 6D data; 0.7 IoU, not the strict 0.95 reviewed dataset.'))
    print('Packed %d; rejected %d' % (len(packed), len(rejected)))
    if not packed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
