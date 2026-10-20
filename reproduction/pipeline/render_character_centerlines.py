"""Render all single-path stroke centerlines of each character on one canvas."""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import arrays, digest, gray, output_dir, png, preview, records, require, write_json, write_records


def _stroke_color(index, count):
    hsv = np.uint8([[[int(round(179 * index / max(1, count))), 210, 220]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def render_centerlines(rows, output, line_width=2, label=True):
    rows = sorted(rows, key=lambda row: row['stroke_index'])
    first = rows[0]
    size = int(first.get('canvas', 256))
    require(size > 0, 'invalid canvas size')
    canvas = np.full((size, size, 3), 255, np.uint8)
    colored = canvas.copy()
    target = gray(first['image_path']) < 128
    overlay = np.full((size, size, 3), 255, np.uint8)
    overlay[target] = (232, 232, 232)
    metadata_rows = []
    count = len(rows)
    for row in rows:
        z = arrays(row['trajectory_path'])
        points = np.asarray(z['points'], dtype=np.float32)
        offsets = np.asarray(z['subpath_offsets'], dtype=np.int64)
        require(len(offsets) == 2, 'Expected exactly one subpath: ' + row['stroke_id'])
        xy = np.rint(points[:, :2]).astype(np.int32)
        require(len(xy) >= 2, 'Centerline has fewer than two points: ' + row['stroke_id'])
        stroke_index = int(row['stroke_index'])
        color = _stroke_color(stroke_index, count)
        cv2.polylines(canvas, [xy.reshape(-1, 1, 2)], False, (25, 25, 25), line_width, cv2.LINE_AA)
        cv2.polylines(colored, [xy.reshape(-1, 1, 2)], False, color, line_width, cv2.LINE_AA)
        cv2.polylines(overlay, [xy.reshape(-1, 1, 2)], False, color, line_width, cv2.LINE_AA)
        start = tuple(int(v) for v in xy[0])
        end = tuple(int(v) for v in xy[-1])
        cv2.circle(colored, start, max(2, line_width + 1), (0, 170, 0), -1, cv2.LINE_AA)
        cv2.circle(colored, end, max(2, line_width + 1), (0, 0, 220), -1, cv2.LINE_AA)
        cv2.circle(overlay, start, max(2, line_width + 1), (0, 170, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, end, max(2, line_width + 1), (0, 0, 220), -1, cv2.LINE_AA)
        if label:
            anchor = (min(size - 1, start[0] + 3), max(12, start[1] - 3))
            text = str(stroke_index)
            cv2.putText(colored, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(colored, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        metadata_rows.append(dict(stroke_id=row['stroke_id'], stroke_index=stroke_index,
                                  trajectory_sha256=digest(row['trajectory_path']),
                                  point_count=int(len(points)), start=start, end=end,
                                  color_bgr=list(color)))
    directory = output / first['sample_id']
    directory.mkdir()
    centerline_path = png(directory / 'centerline.png', canvas)
    order_path = png(directory / 'centerline_order.png', colored)
    overlay_path = png(directory / 'centerline_overlay.png', overlay)
    item = dict(schema='chtr-character-centerlines-v1', sample_id=first['sample_id'],
                character=first['character'], stroke_count=count,
                centerline_path=centerline_path, order_path=order_path,
                overlay_path=overlay_path, image_path=first['image_path'],
                preview_path=order_path, strokes=metadata_rows,
                source_manifest_row_count=len(rows), exploratory=True)
    write_json(directory / 'metadata.json', item)
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='single-path stroke manifest')
    parser.add_argument('--output', required=True)
    parser.add_argument('--line-width', type=int, default=2)
    parser.add_argument('--no-label', action='store_true')
    args = parser.parse_args()
    require(args.line_width > 0, 'line-width must be positive')
    groups = defaultdict(list)
    for row in records(args.manifest):
        groups[row['sample_id']].append(row)
    require(groups, 'empty manifest')
    out = output_dir(args.output)
    result = []
    for sample_id, rows in sorted(groups.items()):
        result.append(render_centerlines(rows, out, args.line_width, not args.no_label))
    write_records(out / 'manifest.jsonl', result)
    write_json(out / 'summary.json', dict(schema='chtr-character-centerlines-v1',
        samples=len(result), total_strokes=sum(row['stroke_count'] for row in result),
        line_width=args.line_width, labels=not args.no_label, exploratory=True))
    preview(out, result, 'All stroke centerlines per character')
    print('Rendered %d characters; %d strokes' %
          (len(result), sum(row['stroke_count'] for row in result)))


if __name__ == '__main__':
    main()
