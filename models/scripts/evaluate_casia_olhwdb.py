"""Image-only two-stage evaluation on official POT trajectories: glyph mIoU, exact DTW and FID."""
import argparse
import itertools
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.datasets.casia_pot import iter_pot, normalized_strokes
from models.networks.two_stage import recover_trajectories
from models.training.ordered_stroke import load_model as load_stroke
from models.training.single_stroke import load_model as load_point, local_path
from models.training.single_stroke_evaluation import raster
from models.training.casia_metrics import dtw_xy, DTW_BACKEND, FeatureMoments
from models.scripts.infer_two_stage_batch import atomic_json, sha256, reconstruction_metrics, save_csv
from models.scripts.visualize_two_stage import render_outputs


def fixed_raster(strokes, width):
    result = np.zeros((256, 256), bool)
    for points in strokes:
        if len(points):
            result |= raster(np.column_stack([points[:, :2], np.full(len(points), width)]))
    return result


def pack_strokes(strokes, columns=2):
    points = np.concatenate(strokes) if strokes else np.empty((0, columns), np.float32)
    return points, np.r_[0, np.cumsum([len(p) for p in strokes])].astype(np.int64)


def aggregate(rows):
    if not rows:
        return dict(images=0)
    valid_dtw = [r for r in rows if r['dtw_status'] == 'ok']
    times = np.array([r['inference_seconds'] for r in rows])
    return dict(images=len(rows), miou=float(np.mean([r['foreground_iou'] for r in rows])),
                binary_class_miou=float(np.mean([r['binary_class_miou'] for r in rows])),
                miou_predicted_width=float(np.mean([r['predicted_width_iou'] for r in rows])),
                dtw_sum_px_mean=float(np.mean([r['dtw_sum_px'] for r in valid_dtw])) if valid_dtw else None,
                dtw_path_mean_px_mean=float(np.mean([r['dtw_path_mean_px'] for r in valid_dtw])) if valid_dtw else None,
                dtw_valid_images=len(valid_dtw), dtw_undefined_images=len(rows)-len(valid_dtw),
                empty_predictions=sum(r['predicted_strokes'] == 0 for r in rows),
                mean_predicted_strokes=float(np.mean([r['predicted_strokes'] for r in rows])),
                mean_gt_strokes=float(np.mean([r['gt_strokes'] for r in rows])),
                stroke_count_mae=float(np.mean([abs(r['predicted_strokes']-r['gt_strokes']) for r in rows])),
                stroke_cap_image_rate=float(np.mean([r['stroke_truncated'] for r in rows])),
                point_cap_image_rate=float(np.mean([r['point_truncations'] > 0 for r in rows])),
                timing_seconds=dict(mean=float(times.mean()), median=float(np.median(times)),
                                    p95=float(np.percentile(times, 95)), total=float(times.sum())))


def calculate_fid(rows, output, device, batch_size):
    from pytorch_fid.inception import InceptionV3, FID_WEIGHTS_URL
    from pytorch_fid.fid_score import calculate_frechet_distance
    if len(rows) < 2:
        return dict(value=None, samples=len(rows), status='undefined_fewer_than_two_images')
    # pytorch-fid's internal loader does not hash-check downloads by default.
    weight_path = Path(torch.hub.get_dir())/'checkpoints'/FID_WEIGHTS_URL.rsplit('/', 1)[-1]
    expected_prefix = weight_path.stem.rsplit('-', 1)[-1]
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    if weight_path.exists() and not sha256(weight_path).startswith(expected_prefix):
        invalid = weight_path.with_name(weight_path.name+'.invalid-'+uuid4().hex[:8])
        weight_path.rename(invalid)
        print('Preserved invalid FID cache as '+str(invalid), flush=True)
    if not weight_path.exists():
        temporary = weight_path.with_suffix('.pth.download')
        torch.hub.download_url_to_file(FID_WEIGHTS_URL, str(temporary), hash_prefix=expected_prefix)
        temporary.replace(weight_path)
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    real, fake = FeatureMoments(), FeatureMoments()
    for offset in range(0, len(rows), batch_size):
        reference, generated = [], []
        for row in rows[offset:offset+batch_size]:
            with np.load(output/row['artifact'], allow_pickle=False) as z:
                reference.append((~z['reference_mask']).astype(np.float32))
                generated.append((~z['prediction_mask']).astype(np.float32))
        images = np.stack(reference+generated)
        tensor = torch.from_numpy(images[:, None]).repeat(1, 3, 1, 1).to(device)
        with torch.no_grad():
            features = model(tensor)[0].flatten(1).cpu().numpy()
        real.update(features[:len(reference)])
        fake.update(features[len(reference):])
        if offset % (batch_size*20) == 0:
            print(f'FID features {offset+len(reference)}/{len(rows)}', flush=True)
    for label, moments in [('real', real), ('prediction', fake)]:
        np.savez(output/f'fid_{label}_stats.npz', mu=moments.mean, sigma=moments.covariance(), count=moments.count)
    fid = float(calculate_frechet_distance(real.mean, real.covariance(), fake.mean, fake.covariance()))
    if not np.isfinite(fid):
        raise ValueError('Nonfinite FID')
    return dict(value=fid, samples=len(rows), dimensions=2048, extractor='pytorch-fid InceptionV3 FID weights',
                package_version=version('pytorch-fid'),
                weights_sha256=sha256(weight_path),
                status='complete', small_sample_warning=len(rows) <= 2048,
                rendering='same fixed width for GT and predicted XY; grayscale repeated to RGB; package resize/normalize')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pot-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stroke-checkpoint', default='models/checkpoints/ordered_stroke.pt')
    parser.add_argument('--point-checkpoint', default='models/checkpoints/single_stroke.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--point-batch-size', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--line-width', type=int, default=8, help='Synthetic pixel diameter, not a measured CASIA width')
    parser.add_argument('--margin', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='First N eligible records in sorted POT order; 0=all')
    parser.add_argument('--include-symbols', action='store_true', help='Default: retain only decoded CJK characters')
    parser.add_argument('--visualize-count', type=int, default=30)
    parser.add_argument('--skip-fid', action='store_true', help='Explicitly skip FID for connection tests')
    parser.add_argument('--fid-batch-size', type=int, default=32)
    parser.add_argument('--torch-home', default='data/CASIA-OLHWDB/.torch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint-every', type=int, default=1000, help='Recompute full aggregate every N images; lightweight progress every 15 seconds')
    args = parser.parse_args()
    if min(args.threads, args.point_batch_size, args.warmup, args.fid_batch_size, args.checkpoint_every) < 1 or not 1 <= args.line_width <= 64 or not 0 <= args.margin < 128 or min(args.limit, args.visualize_count) < 0:
        parser.error('Invalid thread/batch/warmup/width/margin/limit settings')
    if not args.skip_fid:
        try:
            import pytorch_fid
            import scipy
        except ImportError:
            parser.error('Install models/requirements-casia-eval.txt, or explicitly use --skip-fid for a smoke test')
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    os.environ['TORCH_HOME'] = str(local_path(args.torch_home))
    device = torch.device(args.device)
    pot_root, output = local_path(args.pot_root).resolve(), local_path(args.output).resolve()
    files = sorted(pot_root.rglob('*.pot'))
    if not files:
        parser.error('No POT files; MPF/GNT/WPTT files are not supported by this isolated-character evaluator')
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Output must be empty, or use --resume')
    output.mkdir(parents=True, exist_ok=True)
    config = dict(protocol_version=1, pot_files=[dict(path=str(p), sha256=sha256(p)) for p in files],
                  checkpoints=[dict(path=str(local_path(p).resolve()), sha256=sha256(local_path(p)))
                               for p in [args.stroke_checkpoint, args.point_checkpoint]],
                  input_size=256, line_width=args.line_width, margin=args.margin, limit=args.limit,
                  include_symbols=args.include_symbols, device=str(device), threads=args.threads,
                  point_batch_size=args.point_batch_size, warmup=args.warmup,
                  visualize_count=args.visualize_count,
                  protocol=dict(miou='macro mean of whole-character foreground IoU, BOTH rendered at fixed width',
                                binary_class_miou='macro mean over foreground/background IoU',
                                dtw='sum of Euclidean XY costs along exact min-cost path; original point order, no resampling, no pen-up coordinate insertion',
                                dtw_path_mean='same optimal path cost divided by path length; not PEN-Net LDTW',
                                fid='dataset-level 2048-D pytorch-fid features, same fixed-width rendering as mIoU',
                                scope='local 256x256 external evaluation; not a claim of matching any published split/render/sampling protocol',
                                width='POT has XY and pen-up boundaries, no measured width/pressure/time; diameter is synthetic'))
    config_path = output/'run_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            parser.error('Input/checkpoint/config differs from previous run; choose new output')
    else:
        atomic_json(config_path, config)
    def samples():
        emitted = 0
        for path in files:
            for sample in iter_pot(path):
                char = sample['char']
                if not args.include_symbols and not (char and len(char) == 1 and '\u3400' <= char <= '\u9fff'):
                    continue
                relative = path.relative_to(pot_root).with_suffix('')/f'{sample["index"]:06d}_{sample["tag_code"]:04x}'
                sample['artifact'] = 'predictions/'+relative.as_posix()+'.npz'
                sample['example'] = 'examples/'+relative.as_posix()
                yield sample
                emitted += 1
                if args.limit and emitted >= args.limit:
                    return
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    def input_for(sample):
        strokes, transform = normalized_strokes(sample['strokes'], margin=args.margin)
        mask = fixed_raster(strokes, args.line_width)
        image = np.where(mask, 0, 255).astype(np.uint8)
        inputs = torch.from_numpy(image.astype(np.float32)[None, None]/255.).to(device)
        return strokes, transform, mask, image, inputs
    # JSONL is the committed per-image record; incomplete last writes are discarded on resume.
    rows = {}
    journal = output/'per_image.jsonl'
    if journal.exists():
        position = 0
        with journal.open('rb') as stream:
            while True:
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    break
                row = json.loads(line)
                if not (output/row['artifact']).is_file():
                    raise ValueError('Committed prediction artifact is missing: '+row['artifact'])
                rows[row['artifact']] = row
                position = stream.tell()
        with journal.open('r+b') as stream:
            stream.truncate(position)
    fid_status = dict(value=None, status='pending' if not args.skip_fid else 'explicitly_skipped')
    old_summary = json.loads((output/'summary.json').read_text()) if (output/'summary.json').exists() else {}
    sessions = old_summary.get('sessions', [])
    session = dict(warmup_runs=0, warmup_seconds=0., model_load_seconds=0., images=0)
    def snapshot(status):
        current = list(rows.values())
        atomic_json(output/'summary.json', dict(status=status, metrics=aggregate(current), fid=fid_status,
                    sessions=sessions, dtw_backend=DTW_BACKEND, protocol=config['protocol'],
                    per_image_records='per_image.jsonl', per_image_table='per_image.csv',
                    inference_timing='CUDA synchronized, excludes model load, warmup, rasterization, transfer, metrics, saving, visualization and FID'))
    first = list(itertools.islice((s for s in samples() if s['artifact'] not in rows), args.warmup))
    if first:
        sessions.append(session)
        started = perf_counter()
        stroke, stroke_ck = load_stroke(args.stroke_checkpoint, args.device)
        point, point_ck = load_point(local_path(args.point_checkpoint), args.device)
        stroke.eval(); point.eval(); sync()
        session['model_load_seconds'] = perf_counter()-started
        @torch.inference_mode()
        def infer(inputs):
            return recover_trajectories(stroke, point, inputs,
                stroke_threshold=stroke_ck['config']['validity_threshold'],
                point_threshold=point_ck['config'].get('validity_threshold', .5),
                point_batch_size=args.point_batch_size)[0]
        started = perf_counter()
        for i in range(args.warmup):
            *_, inputs = input_for(first[i % len(first)])
            result = infer(inputs)
            sync()
            del result, inputs
        session.update(warmup_runs=args.warmup, warmup_seconds=perf_counter()-started)
        snapshot('running')
        print(f'Warmup {args.warmup} runs excluded; resumed {len(rows)} images; DTW: {DTW_BACKEND}', flush=True)
        last_progress = perf_counter()
        sum_iou = sum(r['foreground_iou'] for r in rows.values())
        with journal.open('a', encoding='utf-8') as stream:
            for sample in samples():
                if sample['artifact'] in rows:
                    continue
                gt_strokes, transform, reference, image, inputs = input_for(sample)
                sync(); started = perf_counter()
                result = infer(inputs)
                sync(); elapsed = perf_counter()-started
                predicted = [r['points'].cpu().numpy()*np.array([255, 255, 256], np.float32) for r in result['strokes']]
                pred_mask = fixed_raster(predicted, args.line_width)
                width_mask = np.zeros_like(reference)
                for path in predicted:
                    width_mask |= raster(path)
                overlap = reconstruction_metrics(reference, pred_mask)
                background = reconstruction_metrics(~reference, ~pred_mask)['iou']
                gt_points, gt_offsets = pack_strokes(gt_strokes)
                pred_points, pred_offsets = pack_strokes(predicted, columns=3)
                row = dict(artifact=sample['artifact'], pot_file=sample['pot_file'], record_index=sample['index'],
                           byte_offset=sample['byte_offset'], writer_id=sample['writer_id'], char=sample['char'], tag_code=sample['tag_code'],
                           label_error=sample['label_error'], inference_seconds=elapsed,
                           foreground_iou=overlap['iou'], binary_class_miou=(overlap['iou']+background)/2,
                           predicted_width_iou=reconstruction_metrics(reference, width_mask)['iou'],
                           gt_strokes=len(gt_strokes), predicted_strokes=len(predicted),
                           gt_points=len(gt_points), predicted_points=len(pred_points),
                           stroke_truncated=bool(result['instances']['truncated']),
                           point_truncations=sum(bool(r['truncated']) for r in result['strokes']),
                           transform=transform, **dtw_xy(gt_points, pred_points))
                target = output/sample['artifact']
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.with_suffix('.npz.tmp').open('wb') as artifact:
                    np.savez_compressed(artifact, points=pred_points, stroke_offsets=pred_offsets,
                                        gt_points=gt_points, gt_stroke_offsets=gt_offsets,
                                        reference_mask=reference, prediction_mask=pred_mask, image_size=256)
                target.with_suffix('.npz.tmp').replace(target)
                if len(rows) < args.visualize_count:
                    example = output/sample['example']
                    example.mkdir(parents=True, exist_ok=True)
                    render_outputs(image, pred_points, pred_offsets,
                                   result['instances']['masks'].cpu().numpy(), example)
                    cv2.imwrite(str(example/'input.png'), image)
                    cv2.imwrite(str(example/'fixed_width_reconstruction.png'), np.where(pred_mask, 0, 255).astype(np.uint8))
                    np.savez_compressed(example/'prediction.npz', points=pred_points, stroke_offsets=pred_offsets,
                                        masks=result['instances']['masks'].cpu().numpy() >= .5,
                                        starts=result['instances']['starts'].cpu().numpy()*[255, 255, 256],
                                        boxes=result['instances']['boxes'].cpu().numpy()*255, image_size=256)
                    row['example_directory'] = sample['example']
                    atomic_json(example/'summary.json', dict(row, image=str(example/'input.png')))
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); stream.flush()
                rows[sample['artifact']] = row
                sum_iou += row['foreground_iou']
                session['images'] += 1
                del result, inputs
                if len(rows) % args.checkpoint_every == 0:
                    snapshot('running')
                if len(rows) % 100 == 0 or perf_counter()-last_progress > 15:
                    atomic_json(output/'progress.json', dict(images=len(rows), miou=sum_iou/len(rows), status='running'))
                    print(f'{len(rows)} images; mIoU={sum_iou/len(rows):.4f}', flush=True)
                    last_progress = perf_counter()
        del stroke, point, stroke_ck, point_ck
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if not rows:
        parser.error('No eligible Chinese samples; check labels or use --include-symbols')
    save_csv(output/'per_image.csv', list(rows.values()))
    snapshot('computing_fid' if not args.skip_fid else 'complete')
    if not args.skip_fid:
        started = perf_counter()
        fid_status = calculate_fid(list(rows.values()), output, device, args.fid_batch_size)
        fid_status['seconds'] = perf_counter()-started
    snapshot('complete')
    atomic_json(output/'progress.json', dict(images=len(rows), miou=aggregate(list(rows.values()))['miou'], status='complete'))
    print(json.dumps(dict(metrics=aggregate(list(rows.values())), fid=fid_status), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
