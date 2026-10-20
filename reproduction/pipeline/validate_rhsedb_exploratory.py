"""Validate experimental RHSEDB image/6D pairs and original per-stroke masks."""
import argparse
from pathlib import Path

import numpy as np

from common import arrays, digest, gray, iou, records, render, require, states, validate_points, write_json


def validate(rows):
    errors = []
    seen = set()
    for row in rows:
        try:
            sid = row['sample_id']
            require(sid not in seen, 'duplicate sample ID')
            seen.add(sid)
            require(row['schema'] == 'rhsedb-6d-exploratory-v1', 'unexpected schema')
            require(row['writer_status'] == 'unknown_in_local_npz' and row['dataset_split'] == 'unassigned', 'invented writer/split')
            require(row['manual_review_status'] == 'not_performed', 'unexpected review claim')
            require(row['qc_threshold'] == .7, 'unexpected QC threshold')
            for key, hash_key in (('npz_path', 'npz_sha256'), ('image_path', 'image_sha256'),
                                  ('source_npz_path', 'source_npz_sha256')):
                require(digest(row[key]) == row[hash_key], 'hash mismatch: ' + key)
            z = arrays(row['npz_path'])
            points, offsets, stroke_offsets = z['points'], z['subpath_offsets'], z['stroke_offsets']
            validate_points(points, offsets)
            count = row['stroke_count']
            require(z['stroke_types'].shape == (count,) and stroke_offsets.shape == (count + 1,), 'stroke metadata shape')
            require(stroke_offsets[0] == 0 and stroke_offsets[-1] == len(points) and np.all(np.diff(stroke_offsets) > 0), 'stroke offsets')
            require(set(stroke_offsets) <= set(offsets), 'stroke/subpath boundary mismatch')
            with np.load(row['source_npz_path'], allow_pickle=False) as original:
                require(str(original['name']) == row['character'], 'character mismatch')
                labels = original['stroke_label']
                masks = original['target_single_image'].astype(bool)
                target = original['target_image'][0].astype(bool)
            require(np.array_equal(z['stroke_types'], labels), 'stroke labels changed')
            require(masks.shape == (count, 256, 256), 'source mask shape')
            require(np.array_equal(gray(row['image_path']) < 128, target), 'image differs from source')
            for index in range(count):
                low, high = stroke_offsets[index:index + 2]
                sub = offsets[(offsets >= low) & (offsets <= high)] - low
                part = states(points[low:high], sub)
                score = iou(render(part, sub) >= 128, masks[index])
                require(score >= .7 and abs(score - row['stroke_scores'][index]['target_iou']) < 1e-8,
                        'stroke IoU mismatch/below threshold: %d' % index)
                require(row['stroke_scores'][index]['upstream_iou'] >= .7 and
                        row['stroke_scores'][index]['native_replay_iou'] >= .999, 'upstream QC below threshold')
            whole = iou(render(points, offsets) >= 128, target)
            require(whole >= .7 and abs(whole - row['whole_iou']) < 1e-8, 'whole-character IoU mismatch/below threshold')
        except (KeyError, ValueError, OSError) as exc:
            errors.append(row.get('sample_id', '?') + ': ' + str(exc))
    return dict(valid=bool(rows) and not errors, samples=len(rows), errors=errors,
                minimum_iou=.7, writer_split_available=False, manual_review_performed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--report', required=True)
    args = p.parse_args()
    result = validate(records(args.manifest))
    write_json(args.report, result)
    print(result)
    if not result['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
