"""Export automatically threshold-selected cleaned masks for exploratory VS processing."""
import argparse
import json

import cv2
import numpy as np

from common import arrays, digest, output_dir, png, records, require, write_json, write_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--cleanup-report', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--iou', type=float, default=.87)
    a = p.parse_args()
    require(0 < a.iou <= 1, 'invalid IoU threshold')
    rows = records(a.manifest)
    report = json.load(open(a.cleanup_report, encoding='utf-8'))
    reports = {row['sample_id']: row for row in report['samples']}
    out = output_dir(a.output); result=[]; selected=[]; rejected=[]
    for row in rows:
        item = reports[row['sample_id']]
        if item['cleanup_status'] != 'clean_candidate' or item['after']['iou'] < a.iou:
            rejected.append(dict(sample_id=row['sample_id'], character=row['character'],
                                 cleanup_status=item['cleanup_status'], iou=item['after']['iou']))
            continue
        z = arrays(row['extracted_path'])
        masks = z['extractnet_strokes'].astype(bool); labels=z['stroke_labels'].astype(np.int64)
        require(len(masks) == len(labels) and len(masks) > 0, 'mask/label mismatch')
        for index, mask in enumerate(masks):
            components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1
            require(components == 1, 'cleaned mask is not single-component')
            stroke_id = row['sample_id'] + '_stroke%03d' % index
            path = png(out / (stroke_id + '.png'), (~mask).astype(np.uint8) * 255)
            result.append(dict(row, stroke_id=stroke_id, stroke_index=index,
                stroke_label=int(labels[index]), expected_strokes=len(masks), input_png=path,
                input_sha256=digest(path), selection_iou=float(item['after']['iou']),
                selection_threshold=a.iou, review_status='auto_selected_exploratory',
                selection_basis='clean_candidate_and_whole_union_iou_threshold'))
        selected.append(dict(sample_id=row['sample_id'], character=row['character'],
                             strokes=len(masks), iou=item['after']['iou']))
    write_records(out / 'manifest.jsonl', result)
    write_records(out / 'rejected.jsonl', rejected)
    write_json(out / 'summary.json', dict(schema='auto-selected-vs-inputs-v1',
        threshold=a.iou, selected_samples=len(selected), selected_strokes=len(result),
        rejected_samples=len(rejected), selected=selected,
        note='Exploratory automatic selection; no human mask approval is claimed.'))


if __name__ == '__main__':
    main()
