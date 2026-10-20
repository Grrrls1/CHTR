"""Compare original VS candidates with the exploratory single-path route."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import arrays, output_dir, records, require, write_json


def metrics(row):
    z = arrays(row['trajectory_path'])
    points = z['points']
    offsets = z['subpath_offsets']
    length = 0.0
    for start, end in zip(offsets[:-1], offsets[1:]):
        length += float(np.linalg.norm(np.diff(points[start:end, :2], axis=0), axis=1).sum())
    qc = row.get('qc', {})
    def optional_float(value):
        return None if value is None else float(value)
    return dict(stroke_id=row['stroke_id'], character=row.get('character'),
                stroke_index=row.get('stroke_index'), subpaths=int(len(offsets) - 1),
                iou=float(qc.get('iou', row.get('whole_iou', 0.0))),
                upstream_iou=optional_float(qc.get('upstream_iou')),
                path_length=length, render_area_over_target=float(qc.get('render_area_over_target', np.nan)),
                overdraw_area_ratio=optional_float(qc.get('overdraw_area_ratio')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vs-manifest', required=True)
    parser.add_argument('--single-path-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    vs = {row['stroke_id']: metrics(row) for row in records(args.vs_manifest)}
    single = {row['stroke_id']: metrics(row) for row in records(args.single_path_manifest)}
    require(vs and single and set(vs) == set(single), 'Route manifests must cover the same stroke IDs')
    rows = []
    for stroke_id in sorted(vs):
        left, right = vs[stroke_id], single[stroke_id]
        rows.append(dict(stroke_id=stroke_id, character=left['character'],
                         stroke_index=left['stroke_index'],
                         vs_subpaths=left['subpaths'], single_subpaths=right['subpaths'],
                         subpath_reduction=left['subpaths'] - right['subpaths'],
                         vs_iou=left['iou'], single_iou=right['iou'],
                         iou_delta=right['iou'] - left['iou'],
                         vs_path_length=left['path_length'], single_path_length=right['path_length'],
                         vs_overdraw_area_ratio=left['overdraw_area_ratio'],
                         single_overdraw_area_ratio=right['overdraw_area_ratio']))
    out = output_dir(args.output)
    with (out / 'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    vs_iou = np.asarray([row['vs_iou'] for row in rows])
    single_iou = np.asarray([row['single_iou'] for row in rows])
    vs_subpaths = np.asarray([row['vs_subpaths'] for row in rows])
    single_subpaths = np.asarray([row['single_subpaths'] for row in rows])
    summary = dict(schema='chtr-single-path-route-comparison-v1', strokes=len(rows),
                   original_vs=dict(subpaths_median=float(np.median(vs_subpaths)),
                                    single_subpath_count=int(np.sum(vs_subpaths == 1)),
                                    iou_median=float(np.median(vs_iou)),
                                    iou_pass_087=int(np.sum(vs_iou >= .87))),
                   single_path=dict(subpaths_median=float(np.median(single_subpaths)),
                                    single_subpath_count=int(np.sum(single_subpaths == 1)),
                                    iou_median=float(np.median(single_iou)),
                                    iou_pass_087=int(np.sum(single_iou >= .87))),
                   iou_delta=dict(median=float(np.median(single_iou - vs_iou)),
                                  mean=float(np.mean(single_iou - vs_iou))),
                   note='Exploratory geometric baseline; neither route is human-approved.')
    write_json(out / 'summary.json', summary)
    write_json(out / 'comparison.json', rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
