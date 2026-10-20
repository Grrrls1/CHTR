"""Compare saved StrokeExtraction stages against the normalized target foreground."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def metrics(mask, target):
    mask = mask.astype(bool)
    intersection = int(np.count_nonzero(mask & target))
    predicted = int(np.count_nonzero(mask))
    actual = int(np.count_nonzero(target))
    union = predicted + actual - intersection
    return dict(pixels=predicted, covered_pixels=intersection,
                missed_pixels=actual - intersection,
                extra_pixels=predicted - intersection,
                recall=intersection / actual if actual else 0,
                precision=intersection / predicted if predicted else 0,
                iou=intersection / union if union else 0)


def panel(title, mask, target):
    # Blue: correctly covered target; red: missed target; yellow: extra prediction.
    image = np.full((256, 256, 3), 255, np.uint8)
    image[mask & target] = (200, 85, 35)
    image[target & ~mask] = (40, 40, 225)
    image[mask & ~target] = (40, 190, 230)
    image = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(image, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def inspect(row, reference, output):
    sample = row['sample_id']
    target_image = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
    if target_image is None or target_image.shape != (256, 256):
        raise ValueError('Missing 256x256 target: ' + sample)
    target = target_image < 128
    with np.load(reference.get('reference_path', reference.get('npz_path')), allow_pickle=False) as z:
        original = np.any(z['reference_single_image'] > .5, axis=0)
    with np.load(row['extracted_path'], allow_pickle=False) as z:
        stages = {
            'reference_original': original,
            'reference_after_sdnet': np.any(z['sdnet_transformed_single_strokes'] > .5, axis=0),
            'segnet_any_class': np.any(z['segnet_probabilities'] > .5, axis=0),
            'extractnet_union': np.any(z['extractnet_strokes'] > .5, axis=0),
        }
        transformed_target = z['sdnet_transformed_target'][0] > .5
        strokes = z['extractnet_strokes'] > .5
    result = dict(sample_id=sample, character=row['character'], target_pixels=int(target.sum()),
                  stages={name: metrics(mask, target) for name, mask in stages.items()},
                  transformed_target_vs_original=metrics(transformed_target, target),
                  per_stroke=[dict(index=i, pixels=int(mask.sum()),
                                   components=max(0, cv2.connectedComponents(mask.astype(np.uint8))[0] - 1))
                              for i, mask in enumerate(strokes)])
    tiles = [panel('target', target, target)]
    tiles.extend(panel(name, mask, target) for name, mask in stages.items())
    tiles.append(panel('sdnet_target', transformed_target, target))
    montage = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    cv2.imwrite(str(output / (sample + '.png')), montage)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, help='Pipeline run directories with inputs/ and extracted/')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for run_name in args.runs:
        run = Path(run_name)
        samples = {r['sample_id']: r for r in map(json.loads, (run / 'inputs/samples.jsonl').read_text().splitlines())}
        references = {r['character']: r for r in map(json.loads, (run / 'inputs/references.jsonl').read_text().splitlines())}
        extracted = list(map(json.loads, (run / 'extracted/manifest.jsonl').read_text().splitlines()))
        for row in extracted:
            if row['sample_id'] not in samples:
                raise ValueError('Unknown sample: ' + row['sample_id'])
            reports.append(inspect(row, references[row['character']], output))
    (output / 'report.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
