"""Create target/mask overlays and a compact JSON report for an extraction run."""
import argparse
import colorsys
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import arrays, output_dir, preview, records, require, write_json


def aggregate(values):
    values = [float(value) for value in values]
    if not values:
        return None
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                minimum=float(np.min(values)), maximum=float(np.max(values)))


def mask_diagnostics(masks):
    masks = masks.astype(bool)
    union = np.any(masks, axis=0)
    multiplicity = masks.sum(axis=0)
    components, substantial = [], []
    for mask in masks:
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        sizes = sorted((int(value) for value in stats[1:count, cv2.CC_STAT_AREA]), reverse=True)
        components.append(len(sizes))
        substantial.append(sum(value >= 16 for value in sizes))
    duplicates = []
    for first in range(len(masks)):
        for second in range(first + 1, len(masks)):
            overlap = int(np.count_nonzero(masks[first] & masks[second]))
            smaller = min(int(masks[first].sum()), int(masks[second].sum()))
            if smaller and overlap / smaller >= .5:
                duplicates.append(dict(first=first, second=second,
                    overlap_fraction_of_smaller=float(overlap / smaller)))
    overlap_pixels = int(np.count_nonzero(multiplicity >= 2))
    return dict(mask_count=len(masks), total_components=sum(components),
                fragmented_strokes=sum(value > 1 for value in components),
                substantial_components=sum(substantial),
                substantially_fragmented_strokes=sum(value > 1 for value in substantial),
                overlap_pixels=overlap_pixels,
                overlap_fraction_of_union=float(overlap_pixels / union.sum()) if union.any() else 0.0,
                possible_duplicate_pairs=duplicates)


def make_overlay(image_path, masks, destination):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    require(image is not None and image.shape[:2] == (256, 256), 'invalid target image')
    canvas = image.astype(np.float32)
    count = len(masks)
    colors = [np.asarray(colorsys.hsv_to_rgb(i / max(1, count), .85, 1.0))[::-1] * 255
              for i in range(count)]
    for index, (mask, color) in enumerate(zip(masks.astype(bool), colors)):
        canvas[mask] = .35 * canvas[mask] + .65 * color
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color.tolist(), 1)
        y, x = np.where(mask)
        if len(x):
            cv2.putText(canvas, str(index), (int(np.median(x)), int(np.median(y))),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(index), (int(np.median(x)), int(np.median(y))),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
    require(cv2.imwrite(str(destination), np.clip(canvas, 0, 255).astype(np.uint8)),
            'failed to save overlay')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    args = p.parse_args()
    run = Path(args.run).resolve()
    inputs = {row['sample_id']: row for row in records(run / 'inputs/samples.jsonl')}
    stage_rows = {row['sample_id']: row for row in
                  json.loads((run / 'stage_diagnosis/report.json').read_text(encoding='utf-8'))}
    extracted = {row['sample_id']: row for row in records(run / 'extracted/manifest.jsonl')}
    errors = records(run / 'extracted/errors.jsonl')
    overlay_dir = output_dir(run / 'mask_overlays')
    samples, preview_rows = [], []
    for sample_id, stage in stage_rows.items():
        source = inputs[sample_id]
        row = extracted[sample_id]
        masks = arrays(row['extracted_path'])['extractnet_strokes'].astype(bool)
        overlay_path = overlay_dir / (sample_id + '.png')
        make_overlay(source['image_path'], masks, overlay_path)
        final = stage['stages']['extractnet_union']
        diagnostic = mask_diagnostics(masks)
        item = dict(sample_id=sample_id, character=source['character'],
                    stroke_count=int(source['evaluation_stroke_count']),
                    union_iou=float(final['iou']), union_recall=float(final['recall']),
                    union_precision=float(final['precision']), missed_pixels=int(final['missed_pixels']),
                    extra_pixels=int(final['extra_pixels']), **diagnostic,
                    overlay_path=str(overlay_path),
                    stage_diagnosis_path=str(run / 'stage_diagnosis' / (sample_id + '.png')),
                    mask_review_path=str(run / 'mask_review' / sample_id))
        samples.append(item)
        preview_rows.append(dict(sample_id=sample_id, character=source['character'],
                                 stroke_count=item['stroke_count'], overlay_path=str(overlay_path)))
    samples.sort(key=lambda row: (row['stroke_count'], row['sample_id']))
    preview(overlay_dir, preview_rows, 'Predicted stroke masks over target image')

    buckets = [(1, 5), (6, 10), (11, 15), (16, 20)]
    by_complexity = []
    for low, high in buckets:
        rows = [row for row in samples if low <= row['stroke_count'] <= high]
        by_complexity.append(dict(range=[low, high], successful=len(rows),
            mean_iou=aggregate(row['union_iou'] for row in rows),
            mean_recall=aggregate(row['union_recall'] for row in rows),
            mean_precision=aggregate(row['union_precision'] for row in rows),
            substantially_fragmented_strokes=sum(row['substantially_fragmented_strokes'] for row in rows),
            possible_duplicate_pairs=sum(len(row['possible_duplicate_pairs']) for row in rows)))
    error_by_character = defaultdict(list)
    for error in errors:
        source = inputs.get(error['sample_id'], {})
        error_by_character[source.get('character', '?')].append(error['sample_id'])
    report = dict(schema='strokeextraction-evaluation-v1', run=str(run),
        requested_samples=len(inputs), successful_samples=len(samples), failed_samples=len(errors),
        success_rate=float(len(samples) / len(inputs)) if inputs else 0.0,
        overall=dict(iou=aggregate(row['union_iou'] for row in samples),
                     recall=aggregate(row['union_recall'] for row in samples),
                     precision=aggregate(row['union_precision'] for row in samples),
                     substantially_fragmented_strokes=sum(row['substantially_fragmented_strokes'] for row in samples),
                     overlap_fraction=aggregate(row['overlap_fraction_of_union'] for row in samples),
                     possible_duplicate_pairs=sum(len(row['possible_duplicate_pairs']) for row in samples)),
        by_complexity=by_complexity,
        failures=[dict(error, character=inputs.get(error['sample_id'], {}).get('character'),
                       stroke_count=inputs.get(error['sample_id'], {}).get('evaluation_stroke_count'))
                  for error in errors],
        failures_by_character=dict(error_by_character), samples=samples,
        visualization=dict(mask_overlay_index=str(overlay_dir / 'index.html'),
                           stage_diagnosis=str(run / 'stage_diagnosis'),
                           mask_review_index=str(run / 'mask_review/index.html')))
    write_json(run / 'report.json', report)
    print(json.dumps({key: report[key] for key in
                      ('requested_samples', 'successful_samples', 'failed_samples', 'success_rate')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
