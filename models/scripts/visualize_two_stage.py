"""Render an existing two-stage prediction without loading either model."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.training.single_stroke_evaluation import raster
from models.training.single_stroke import local_path


def panel(image, title):
    result = np.full((288, 256, 3), 255, np.uint8)
    result[32:] = image
    cv2.putText(result, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 30, 30), 1, cv2.LINE_AA)
    return result


def centerline(image, points, color, number):
    if not len(points):
        return
    xy_float = points[:, :2]
    xy = np.rint(xy_float).astype(np.int32)
    scale = max(1, image.shape[1]//256)
    cv2.polylines(image, [xy], False, color, 2*scale, cv2.LINE_AA)
    # Fixed arc-length arrows remain visible even with dense/repeated points.
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy_float, axis=0), axis=1))]
    distance, indices = np.unique(distance, return_index=True)
    if distance[-1] >= 3*scale:
        samples = xy_float[indices]
        arrow_length = min(16.*scale, distance[-1]*.35)
        fractions = [.3, .65, .9] if distance[-1] > 35*scale else [.7]
        for fraction in fractions:
            end = distance[-1]*fraction
            start = max(0., end-arrow_length)
            a = tuple(int(round(np.interp(start, distance, samples[:, j]))) for j in range(2))
            b = tuple(int(round(np.interp(end, distance, samples[:, j]))) for j in range(2))
            cv2.arrowedLine(image, a, b, (255, 255, 255), 4*scale, cv2.LINE_AA, tipLength=.45)
            cv2.arrowedLine(image, a, b, color, 2*scale, cv2.LINE_AA, tipLength=.45)
    cv2.rectangle(image, tuple(xy[-1]-3*scale), tuple(xy[-1]+3*scale), color, scale)
    cv2.circle(image, tuple(xy[0]), 5*scale, (255, 255, 255), -1)
    cv2.circle(image, tuple(xy[0]), 3*scale, color, -1)
    position = tuple(np.clip(xy[0]+[7*scale, -6*scale], [0, 12*scale], [image.shape[1]-20*scale, image.shape[0]-1]))
    cv2.putText(image, str(number), position, cv2.FONT_HERSHEY_SIMPLEX, .55*scale, (255, 255, 255), 3*scale, cv2.LINE_AA)
    cv2.putText(image, str(number), position, cv2.FONT_HERSHEY_SIMPLEX, .55*scale, color, scale, cv2.LINE_AA)


def render_outputs(image, points, offsets, masks, directory):
    original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    count = len(offsets)-1
    if masks.shape != (count, 256, 256):
        raise ValueError('mask count or canvas does not match trajectories')
    overlay = original.copy()
    lines = original.copy()
    directions = cv2.resize(np.rint(original*.25+255*.75).astype(np.uint8), (768, 768))
    reconstruction = np.full_like(original, 255)
    stroke_panels = []
    for k in range(count):
        hsv = np.uint8([[[int(k*179/max(count, 1)), 220, 205]]])
        color = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
        path = points[offsets[k]:offsets[k+1]]
        mask = masks[k] >= .5
        overlay[mask] = np.rint(overlay[mask]*.45+np.array(color)*.55).astype(np.uint8)
        centerline(lines, path, color, k+1)
        scaled = path.copy()
        scaled[:, :2] *= 3
        centerline(directions, scaled, color, k+1)
        drawn = raster(path) if len(path) else np.zeros((256, 256), bool)
        reconstruction[drawn] = color
        isolated = np.where(mask[:, :, None], original, 255).astype(np.uint8)
        stroke_line = isolated.copy()
        centerline(stroke_line, path, color, k+1)
        stroke_width = np.full_like(original, 255)
        stroke_width[drawn] = color
        stroke_panels.append(np.concatenate([
            panel(isolated, f'Stroke {k+1}: masked input'),
            panel(stroke_line, f'Stroke {k+1}: XY ({len(path)} pts)'),
            panel(stroke_width, f'Stroke {k+1}: XYW rendering')], axis=1))
    overview = np.concatenate([panel(original, 'Input'), panel(overlay, 'Predicted masks (>=0.5)'),
                               panel(lines, 'XY: dot=start, arrows=order'),
                               panel(reconstruction, 'XYW width rendering')], axis=1)
    output = directory
    output.mkdir(exist_ok=True)
    if not cv2.imwrite(str(output/'overview.png'), overview):
        raise OSError('Failed to save overview')
    direction_panel = np.full((812, 768, 3), 255, np.uint8)
    direction_panel[44:] = directions
    cv2.putText(direction_panel, 'Predicted order: number | START: dot | END: square | arrows: direction',
                (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output/'directions.png'), direction_panel):
        raise OSError('Failed to save directions')
    if stroke_panels and not cv2.imwrite(str(output/'strokes.png'), np.concatenate(stroke_panels, axis=0)):
        raise OSError('Failed to save stroke panels')
    return output


def render_directory(directory):
    summary = json.loads((directory/'summary.json').read_text())
    image = cv2.imread(str(local_path(summary.get('input_image', summary['image']))), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (256, 256):
        raise ValueError('summary image must exist and be 256x256')
    with np.load(directory/'prediction.npz', allow_pickle=False) as prediction:
        return render_outputs(image, prediction['points'], prediction['stroke_offsets'], prediction['masks'], directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--prediction-dir')
    inputs.add_argument('--batch-root', help='Render completed records from a batch summary, without rerunning inference')
    parser.add_argument('--script', choices=['all', 'kaishu'], default='all')
    args = parser.parse_args()
    if args.prediction_dir:
        print(render_directory(local_path(args.prediction_dir)))
    else:
        root = local_path(args.batch_root)
        summary = json.loads((root/'summary.json').read_text())
        rows = [r for r in summary['per_image'] if r['status'] == 'ok'
                and (args.script == 'all' or r['script_type'] == '楷书')]
        for i, row in enumerate(rows, 1):
            render_directory(root/row['prediction_dir'])
            if i % 20 == 0 or i == len(rows):
                print(f'Rendered {i}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
