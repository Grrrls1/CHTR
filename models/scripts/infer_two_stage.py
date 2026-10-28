"""Image-only two-stage inference on an already normalized 256x256 image."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.networks.two_stage import recover_trajectories
from models.training.ordered_stroke import load_model as load_stroke_model, local_path
from models.training.single_stroke import load_model as load_point_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stroke-checkpoint', required=True)
    parser.add_argument('--point-checkpoint', required=True)
    parser.add_argument('--image', required=True, help='normalized 256x256 grayscale, same image_key coordinate system')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--point-batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.threads < 1 or args.point_batch_size < 1:
        parser.error('threads and point-batch-size must be positive')
    torch.set_num_threads(args.threads)
    image = cv2.imread(str(local_path(args.image)), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (256, 256):
        parser.error('image must already use the normalized 256x256 canvas; no implicit crop/resize')
    output = local_path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error('output must be an empty directory')
    stroke, stroke_ck = load_stroke_model(args.stroke_checkpoint, args.device)
    point, point_ck = load_point_model(local_path(args.point_checkpoint), args.device)
    stroke.eval()
    point.eval()
    images = torch.from_numpy(image.astype(np.float32)[None, None]/255.).to(args.device)
    result = recover_trajectories(stroke, point, images,
        stroke_threshold=stroke_ck['config']['validity_threshold'],
        point_threshold=point_ck['config'].get('validity_threshold', .5),
        point_batch_size=args.point_batch_size)[0]
    trajectories = [r['points'].cpu().numpy()*np.array([255, 255, 256], np.float32) for r in result['strokes']]
    offsets = np.r_[0, np.cumsum([len(p) for p in trajectories])].astype(np.int64)
    points = np.concatenate(trajectories) if trajectories else np.empty((0, 3), np.float32)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'prediction.npz', points=points, stroke_offsets=offsets,
                        masks=result['instances']['masks'].cpu().numpy(),
                        starts=result['instances']['starts'].cpu().numpy()*[255, 255, 256],
                        boxes=result['instances']['boxes'].cpu().numpy()*255, image_size=256)
    report = dict(image=str(local_path(args.image)), stroke_count=len(trajectories),
                  stroke_checkpoint=dict(path=args.stroke_checkpoint, epoch=stroke_ck['epoch'], step=stroke_ck['step']),
                  point_checkpoint=dict(path=args.point_checkpoint, epoch=point_ck.get('epoch'), step=point_ck.get('step')),
                  stroke_stop_reason=result['instances']['stop_reason'],
                  point_stop_reasons=[r['stop_reason'] for r in result['strokes']],
                  point_truncations=sum(r['truncated'] for r in result['strokes']),
                  protocol='masked original + predicted start XYW; no GT input',
                  units='points/starts: pixel XY and pixel diameter; boxes: inclusive pixel-center XYXY; masks: probability')
    (output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
