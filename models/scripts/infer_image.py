"""Single-character photo -> aspect-preserving preprocessing -> existing two-stage inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]


def local(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT/path


def preprocess_image(source, margin=8, crop=None, resize='fit', binarize='none', invert=False):
    if not 0 <= margin < 128:
        raise ValueError('margin must be 0..127')
    if resize not in ('fit', 'pad-only') or binarize not in ('none', 'otsu'):
        raise ValueError('Unknown resize or binarization method')
    with Image.open(source) as original:
        file_size = list(original.size)
        oriented = ImageOps.exif_transpose(original).convert('RGBA')
        background = Image.new('RGBA', oriented.size, (255, 255, 255, 255))
        gray = np.array(Image.alpha_composite(background, oriented).convert('L'))
    height, width = gray.shape
    x, y, w, h = (0, 0, width, height) if crop is None else tuple(crop)
    if min(x, y) < 0 or min(w, h) < 1 or x+w > width or y+h > height:
        raise ValueError('crop X Y WIDTH HEIGHT must lie inside the EXIF-oriented image')
    image = gray[y:y+h, x:x+w].copy()
    if invert:
        image = 255-image
    available = 256-2*margin
    if resize == 'pad-only':
        if max(w, h) > available:
            raise ValueError('pad-only input exceeds canvas minus margins; use --resize fit or reduce --margin')
        nw, nh = w, h
    else:
        scale = available/max(w, h)
        nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    threshold = None
    if binarize == 'otsu':
        threshold, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    left, top = (256-nw)//2, (256-nh)//2
    canvas = np.full((256, 256), 255, np.uint8)
    canvas[top:top+nh, left:left+nw] = image
    sx, sy = nw/w, nh/h
    transform = np.array([[sx, 0, left-sx*x+.5*sx-.5],
                          [0, sy, top-sy*y+.5*sy-.5], [0, 0, 1]], np.float64)
    metadata = dict(source_image=str(Path(source).resolve()), file_size=file_size,
                    source_size=[width, height], source_coordinate_system='EXIF-oriented image pixels',
                    crop_xywh=[x, y, w, h], resize=resize, resized_size=[nw, nh],
                    output_size=[256, 256], scale_xy=[sx, sy], padding=[left, top], margin=margin,
                    inverted=bool(invert), binarization=binarize, otsu_threshold=threshold,
                    source_to_model=transform.tolist(), model_to_source=np.linalg.inv(transform).tolist(),
                    pixel_mapping='(source_xy - crop_origin + 0.5)*scale_xy - 0.5 + padding',
                    notes='Whole ROI resized isotropically before integer-size rounding; original file unchanged; no automatic character detection, lens undistortion, perspective correction or illumination correction')
    return canvas, metadata


def source_xy(points, metadata):
    inverse = np.asarray(metadata['model_to_source'], np.float64)
    homogeneous = np.column_stack([points[:, :2], np.ones(len(points))]) @ inverse.T
    return homogeneous[:, :2]/homogeneous[:, 2:3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--crop', type=int, nargs=4, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'))
    parser.add_argument('--margin', type=int, default=8)
    parser.add_argument('--resize', choices=['fit', 'pad-only'], default='fit')
    parser.add_argument('--binarize', choices=['none', 'otsu'], default='none')
    parser.add_argument('--invert', action='store_true')
    parser.add_argument('--preprocess-only', action='store_true')
    parser.add_argument('--stroke-checkpoint', default='models/checkpoints/ordered_stroke.pt')
    parser.add_argument('--point-checkpoint', default='models/checkpoints/single_stroke.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--point-batch-size', type=int, default=8)
    args = parser.parse_args()
    output = local(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('output must be an empty/new directory')
    started = perf_counter()
    try:
        image, metadata = preprocess_image(local(args.image), args.margin, args.crop,
                                           args.resize, args.binarize, args.invert)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    preprocess_seconds = perf_counter()-started
    process_seconds = 0.
    if not args.preprocess_only:
        with tempfile.TemporaryDirectory(prefix='chtr-image-') as folder:
            temporary = Path(folder)/'input.png'
            if not cv2.imwrite(str(temporary), image):
                raise OSError('Failed to write temporary input')
            command = [sys.executable, str(ROOT/'models/scripts/infer_two_stage.py'),
                       '--image', str(temporary), '--output', str(output), '--visualize',
                       '--stroke-checkpoint', str(local(args.stroke_checkpoint)),
                       '--point-checkpoint', str(local(args.point_checkpoint)),
                       '--device', args.device, '--threads', str(args.threads),
                       '--point-batch-size', str(args.point_batch_size)]
            inference_started = perf_counter()
            try:
                # Existing CLI stays unchanged; reserve its output directory until it finishes.
                subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
            finally:
                output.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output/'input.png'), image):
                    raise OSError('Failed to preserve preprocessed input')
                (output/'preprocess.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
            process_seconds = perf_counter()-inference_started
        with np.load(output/'prediction.npz', allow_pickle=False) as prediction:
            xy = source_xy(prediction['points'], metadata)
            offsets = prediction['stroke_offsets'].copy()
        np.savez_compressed(output/'prediction_source_xy.npz', points_xy=xy, stroke_offsets=offsets,
                            image_size=np.asarray(metadata['source_size']))
        report = json.loads((output/'summary.json').read_text())
        report.update(image=str(output/'input.png'), source_image=metadata['source_image'],
                      preprocessing='preprocess.json', source_xy_prediction='prediction_source_xy.npz',
                      source_xy_units='EXIF-oriented image pixels, NOT robot/world coordinates; no physical width or pressure',
                      pipeline_seconds=dict(preprocessing=preprocess_seconds, inference_process=process_seconds,
                                            total_before_summary_write=perf_counter()-started))
        (output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    else:
        output.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output/'input.png'), image):
            raise OSError('Failed to write input')
        (output/'preprocess.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), preprocess_only=args.preprocess_only,
                          input=str(output/'input.png'), preprocessing=str(output/'preprocess.json')),
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
