"""Run CCSE on saved calligraphy samples and compare masks with StrokeExtraction.

Per-stroke ground truth is unavailable for these samples. Fragmentation and
overlap are diagnostics, not semantic per-stroke accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stats(masks, target):
    masks = masks.astype(bool)
    union = np.any(masks, axis=0) if len(masks) else np.zeros_like(target)
    covered = int(np.count_nonzero(union & target))
    missed = int(np.count_nonzero(target & ~union))
    extra = int(np.count_nonzero(union & ~target))
    multiplicity = masks.sum(axis=0)
    def component_profile(mask):
        n, labels, counts, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        sizes = counts[1:, cv2.CC_STAT_AREA] if n > 1 else []
        return sum(int(size) >= 4 for size in sizes), (float(max(sizes) / sum(sizes)) if len(sizes) else 0.0)

    raw_profiles = [component_profile(mask) for mask in masks]
    clipped_profiles = [component_profile(mask & target) for mask in masks]
    duplicate_pairs = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            overlap = int(np.count_nonzero(masks[i] & masks[j]))
            smaller = min(int(masks[i].sum()), int(masks[j].sum()))
            if smaller and overlap / smaller >= .5:
                duplicate_pairs.append(dict(first=i, second=j,
                                            overlap_fraction_of_smaller=round(overlap / smaller, 4)))
    return dict(count=len(masks), union_iou=covered / (covered + missed + extra),
                union_recall=covered / (covered + missed),
                missed_pixels=missed, extra_pixels=extra,
                multiply_covered_pixels=int(np.count_nonzero(multiplicity >= 2)),
                multiply_covered_fraction_of_union=(float(np.count_nonzero(multiplicity >= 2) / union.sum())
                                                    if union.any() else 0.0),
                component_count_ge4px=[v[0] for v in raw_profiles],
                largest_component_share=[v[1] for v in raw_profiles],
                target_clipped_component_count_ge4px=[v[0] for v in clipped_profiles],
                target_clipped_largest_component_share=[v[1] for v in clipped_profiles],
                possible_duplicate_pairs=duplicate_pairs)


def tile(title, masks, target):
    union = np.any(masks, axis=0) if len(masks) else np.zeros_like(target)
    image = np.full((256, 256, 3), 255, np.uint8)
    image[union & target] = (190, 80, 30)  # blue
    image[target & ~union] = (30, 30, 220)  # red
    image[union & ~target] = (30, 190, 220)  # yellow
    image[masks.sum(axis=0) >= 2] = (170, 40, 170)  # purple
    image = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(image, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def load_rows(runs):
    result = []
    for name in runs:
        run = Path(name)
        result.extend(json.loads(line) for line in (run / 'extracted/manifest.jsonl').read_text().splitlines() if line)
    ids = [row['sample_id'] for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate sample IDs')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--score-thresholds', nargs='+', type=float, default=[.3, .5, .7])
    a = p.parse_args()
    if not a.score_thresholds or any(not 0 < x < 1 for x in a.score_thresholds):
        raise ValueError('Score thresholds must be between 0 and 1')
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    ccse = ROOT / 'CCSE'
    sys.path.insert(0, str(ROOT / 'third_party/detectron2'))
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    model_dir = ccse / 'output/local_handwritten/20260917.193241'
    config_path, weight_path = model_dir / 'config.yaml', model_dir / 'model_final.pth'
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = str(weight_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(min(a.score_thresholds), .05)
    predictor = DefaultPredictor(cfg)
    source = dict(model=str(weight_path), model_sha256=file_hash(weight_path),
                  config=str(config_path), config_sha256=file_hash(config_path),
                  device='cpu', score_thresholds=a.score_thresholds)
    report = []
    for row in load_rows(a.runs):
        sid = row['sample_id']
        image = cv2.imread(row['image_path'])
        if image is None or image.shape[:2] != (256, 256):
            raise ValueError('Expected 256x256 image: ' + sid)
        target = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 128
        instances = predictor(image)['instances'].to('cpu')
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        masks = instances.pred_masks.numpy().astype(bool)
        with np.load(row['extracted_path'], allow_pickle=False) as z:
            previous = z['extractnet_strokes'].astype(bool)
        entries = {}
        montage = [tile('target', np.asarray([target]), target),
                   tile('StrokeExtraction', previous, target)]
        for threshold in a.score_thresholds:
            keep = scores >= threshold
            selected = masks[keep]
            entries[str(threshold)] = dict(stats=stats(selected, target),
                                           scores=np.round(scores[keep], 5).tolist(),
                                           ccse_class_index=classes[keep].tolist())
            montage.append(tile('CCSE score >= %.2f' % threshold, selected, target))
        np.savez_compressed(output / (sid + '.npz'),
                            masks=masks.astype(np.uint8), scores=scores, classes=classes,
                            target=target.astype(np.uint8))
        cv2.imwrite(str(output / (sid + '.png')), np.hstack(montage))
        report.append(dict(sample_id=sid, character=row['character'], image_path=row['image_path'],
                           target_pixels=int(target.sum()),
                           rhsedb_reference_strokes=int(row['expected_strokes']),
                           strokeextraction=stats(previous, target), ccse=entries))
        display_threshold = min(a.score_thresholds, key=lambda value: abs(value - .5))
        print(sid, 'StrokeExtraction', round(report[-1]['strokeextraction']['union_iou'], 3),
              'CCSE %.2f' % display_threshold,
              round(entries[str(display_threshold)]['stats']['union_iou'], 3), flush=True)
    (output / 'report.json').write_text(json.dumps(dict(source=source, samples=report),
                                                   ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
