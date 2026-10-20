"""Prepare CCSE masks for review; never infer stroke order or approve masks."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import digest, png, write_json, write_records


def connected_sizes(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return sorted((int(v) for v in stats[1:n, cv2.CC_STAT_AREA]), reverse=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction', required=True, help='NPZ saved by compare_ccse_stroke_masks.py')
    p.add_argument('--output', required=True)
    p.add_argument('--sample-id', required=True)
    p.add_argument('--character', required=True)
    p.add_argument('--score-threshold', type=float, default=.5)
    p.add_argument('--minimum-vs-pixels', type=int, default=16,
                   help='Keep smaller masks for review, but omit them from the diagnostic VS manifest')
    a = p.parse_args()
    if a.minimum_vs_pixels < 1:
        raise ValueError('minimum-vs-pixels must be positive')
    out = Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be empty: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    with np.load(a.prediction, allow_pickle=False) as z:
        all_masks, scores, classes = z['masks'].astype(bool), z['scores'], z['classes']
        target = z['target'].astype(bool)
    selected = np.flatnonzero(scores >= a.score_threshold)
    masks = all_masks[selected]
    clipped = masks & target
    # Distance inside the *raw* predicted mask keeps a preferred owner at
    # intersections. This is a working partition, not a stroke annotation.
    strengths = np.stack([cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
                          * float(scores[i]) for i, m in zip(selected, masks)])
    owner = np.argmax(strengths, axis=0)
    disjoint = np.stack([clipped[i] & (owner == i) for i in range(len(selected))])
    raw_union = np.any(masks, axis=0)
    remaining = target & ~raw_union
    residual_sizes = connected_sizes(remaining)
    result = []
    contact = []
    for local, original in enumerate(selected):
        sid = '%s_ccse%03d' % (a.sample_id, int(original))
        mask = disjoint[local]
        path = png(out / (sid + '.png'), (~mask).astype(np.uint8) * 255)
        raw_path = png(out / (sid + '_raw.png'), (~masks[local]).astype(np.uint8) * 255)
        clipped_path = png(out / (sid + '_clipped.png'), (~clipped[local]).astype(np.uint8) * 255)
        tile = np.full((256, 256, 3), 255, np.uint8)
        tile[target] = (215, 215, 215)
        tile[mask] = (180, 80, 30)
        tile[clipped[local] & ~mask] = (160, 40, 170)
        tile = cv2.copyMakeBorder(tile, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, '%02d conf %.2f' % (int(original), scores[original]),
                    (5, 19), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)
        contact.append(tile)
        result.append(dict(sample_id=a.sample_id, stroke_id=sid, character=a.character,
                           detection_index=int(original), stroke_index=None,
                           ccse_class_index=int(classes[original]),
                           category_system='CCSE-25', confidence=float(scores[original]),
                           input_png=path, input_sha256=digest(path), raw_png=raw_path,
                           clipped_png=clipped_path, raw_pixels=int(masks[local].sum()),
                           clipped_pixels=int(clipped[local].sum()), assigned_pixels=int(mask.sum()),
                           clipped_component_sizes=connected_sizes(clipped[local]),
                           assigned_component_sizes=connected_sizes(mask),
                           vs_input_status=('candidate' if int(mask.sum()) >= a.minimum_vs_pixels
                                            else 'skip_tiny_after_overlap'),
                           mask_review_status='pending', order_review_status='pending',
                           mask_source='ccse_handwritten_score_filtered_target_clipped_disjoint_candidate'))
    cols = 4
    blank = np.full_like(contact[0], 255)
    rows = [np.hstack((contact[i:i+cols] + [blank] * (cols - len(contact[i:i+cols]))))
            for i in range(0, len(contact), cols)]
    cv2.imwrite(str(out / 'contact_sheet.png'), np.vstack(rows))
    png(out / 'uncovered_target.png', (~remaining).astype(np.uint8) * 255)
    write_records(out / 'manifest.jsonl', result)
    write_records(out / 'vs_ready_manifest.jsonl',
                  [row for row in result if row['vs_input_status'] == 'candidate'])
    write_json(out / 'summary.json', dict(sample_id=a.sample_id, character=a.character,
              prediction_sha256=digest(a.prediction), score_threshold=a.score_threshold,
              selected_masks=len(result), target_pixels=int(target.sum()),
              raw_union_covered_pixels=int((raw_union & target).sum()),
              uncovered_pixels=int(remaining.sum()), uncovered_component_sizes=residual_sizes,
              clipped_overlap_pixels=int(np.count_nonzero(clipped.sum(axis=0) >= 2)),
              disjoint_union_pixels=int(disjoint.any(axis=0).sum()),
              empty_working_masks=[r['detection_index'] for r in result if not r['assigned_pixels']],
              tiny_working_masks=[r['detection_index'] for r in result
                                  if r['vs_input_status'] == 'skip_tiny_after_overlap'],
              vs_candidate_masks=sum(r['vs_input_status'] == 'candidate' for r in result),
              semantic_order='unknown', review_status='pending',
              note='Working masks are pixel-disjoint and target-clipped; no stroke order or label mapping inferred.'))


if __name__ == '__main__':
    main()
