"""Resumable HCSU image-by-image inference, warmup-excluded latency and reconstruction metrics."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.networks.two_stage import recover_trajectories
from models.training.ordered_stroke import load_model as load_stroke_model
from models.training.single_stroke import load_model as load_point_model, local_path
from models.training.single_stroke_evaluation import raster
from models.scripts.visualize_two_stage import render_outputs


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruction_metrics(foreground, reconstruction):
    tp = int((foreground & reconstruction).sum())
    fp = int((~foreground & reconstruction).sum())
    fn = int((foreground & ~reconstruction).sum())
    # Empty prediction against nonempty ink scores zero. Both empty scores one.
    return dict(iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                precision=tp/(tp+fp) if tp+fp else float(not foreground.any()),
                recall=tp/(tp+fn) if tp+fn else 1., tp=tp, fp=fp, fn=fn)


def summarize(rows):
    good = [r for r in rows if r['status'] == 'ok']
    if not good:
        return dict(successful_images=0, failed_images=len(rows))
    times = np.array([r['inference_seconds'] for r in good])
    strokes = sum(r['stroke_count'] for r in good)
    points = sum(r['point_count'] for r in good)
    return dict(successful_images=len(good), failed_images=len(rows)-len(good),
                reconstruction_mean={k: float(np.mean([r['reconstruction'][k] for r in good]))
                                     for k in ['iou', 'dice', 'precision', 'recall']},
                timing_seconds=dict(mean=float(times.mean()), median=float(np.median(times)),
                                    p95=float(np.percentile(times, 95)), min=float(times.min()),
                                    max=float(times.max()), total=float(times.sum())),
                inference_images_per_second=len(good)/float(times.sum()),
                total_strokes=strokes, total_points=points,
                mean_strokes_per_image=strokes/len(good), mean_points_per_image=points/len(good),
                mean_points_per_stroke=points/strokes if strokes else None,
                stroke_cap_image_rate=sum(r['stroke_truncated'] for r in good)/len(good),
                point_cap_image_rate=sum(r['point_truncations'] > 0 for r in good)/len(good),
                point_cap_stroke_rate=sum(r['point_truncations'] for r in good)/strokes if strokes else None,
                any_cap_image_rate=sum(r['stroke_truncated'] or r['point_truncations'] > 0 for r in good)/len(good),
                stroke_stop_reasons=dict(Counter(r['stroke_stop_reason'] for r in good)),
                point_stop_reasons=dict(Counter(s for r in good for s in r['point_stop_reasons'])))


def save_csv(path, rows):
    flat = []
    for row in rows:
        item = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        item.update({'reconstruction_'+k: v for k, v in row.get('reconstruction', {}).items()})
        item['points_per_stroke'] = json.dumps(row.get('points_per_stroke', []))
        flat.append(item)
    fields = list(dict.fromkeys(k for r in flat for k in r))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    temporary.replace(path)


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (256, 256):
        raise ValueError('Expected readable 256x256 image: '+str(path))
    return image


def prepare_input(image, binarize=False):
    if not binarize:
        return image.copy(), None
    threshold, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary, float(threshold)


def save_prediction(directory, result, image, reference_image=None):
    trajectories = [r['points'].cpu().numpy()*np.array([255, 255, 256], np.float32) for r in result['strokes']]
    lengths = [len(p) for p in trajectories]
    points = np.concatenate(trajectories) if trajectories else np.empty((0, 3), np.float32)
    masks = result['instances']['masks'].cpu().numpy() >= .5
    np.savez_compressed(directory/'prediction.npz', points=points,
                        stroke_offsets=np.r_[0, np.cumsum(lengths)].astype(np.int64), masks=masks,
                        starts=result['instances']['starts'].cpu().numpy()*[255, 255, 256],
                        boxes=result['instances']['boxes'].cpu().numpy()*255, image_size=256)
    reconstructed = np.zeros((256, 256), bool)
    for path in trajectories:
        reconstructed |= raster(path)
    reference_image = image if reference_image is None else reference_image
    threshold, binary = cv2.threshold(reference_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    foreground = binary == 0
    original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    comparison = np.full_like(original, 255)
    comparison[foreground & reconstructed] = (40, 140, 40)
    comparison[foreground & ~reconstructed] = (210, 60, 40)  # blue: missed ink
    comparison[~foreground & reconstructed] = (40, 40, 220)  # red: excess ink
    reconstruction_image = np.where(reconstructed, 0, 255).astype(np.uint8)
    for name, array in [('reconstruction.png', reconstruction_image),
                        ('input.png', image),
                        ('comparison.png', np.concatenate([original, cv2.cvtColor(reconstruction_image, cv2.COLOR_GRAY2BGR), comparison], axis=1))]:
        if not cv2.imwrite(str(directory/name), array):
            raise OSError('Failed writing '+str(directory/name))
    render_outputs(image, points, np.r_[0, np.cumsum(lengths)].astype(np.int64), masks, directory)
    return dict(stroke_count=len(lengths), point_count=sum(lengths), points_per_stroke=lengths,
                mean_points_per_stroke=float(np.mean(lengths)) if lengths else None,
                stroke_stop_reason=result['instances']['stop_reason'],
                stroke_truncated=bool(result['instances']['truncated']),
                point_stop_reasons=[r['stop_reason'] for r in result['strokes']],
                point_truncations=sum(bool(r['truncated']) for r in result['strokes']),
                otsu_threshold=float(threshold), foreground_pixels=int(foreground.sum()),
                reconstruction_pixels=int(reconstructed.sum()),
                reconstruction=reconstruction_metrics(foreground, reconstructed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', default='data/HCSU/metadata.csv')
    parser.add_argument('--image-root', default='data/HCSU')
    parser.add_argument('--stroke-checkpoint', default='models/checkpoints/ordered_stroke.pt')
    parser.add_argument('--point-checkpoint', default='models/checkpoints/single_stroke.pt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--point-batch-size', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--source', choices=['all', 'tie', 'bei'], default='all')
    parser.add_argument('--script', choices=['all', 'kaishu'], default='all')
    parser.add_argument('--binarize', action='store_true', help='Per-image Otsu on raw grayscale BEFORE model input; no resizing or morphology')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.threads, args.point_batch_size) < 1 or args.warmup < 1 or args.limit < 0:
        parser.error('threads, point-batch-size, warmup must be positive; limit must be nonnegative')
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    device = torch.device(args.device)
    root = local_path(args.image_root).resolve()
    output = local_path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('output is nonempty; use --resume with identical settings')
    with local_path(args.metadata).open(encoding='utf-8-sig', newline='') as stream:
        selected = [r for r in csv.DictReader(stream)
                    if (args.source == 'all' or r['source_category'].lower() == args.source)
                    and (args.script == 'all' or r['script_type'] == '楷书')]
    selected = selected[:args.limit] if args.limit else selected
    if not selected:
        parser.error('No images selected')
    manifest, seen = [], set()
    for row in selected:
        path = (root/row['image_relative_path']).resolve()
        if root not in path.parents or path in seen:
            parser.error('Invalid or duplicate image path: '+str(path))
        seen.add(path)
        manifest.append(dict(image=str(path), relative=path.relative_to(root).as_posix(),
                             source_category=row['source_category'], script_type=row['script_type'],
                             char=row['char'], calligrapher=row['calligrapher'], sha256=sha256(path)))
    checkpoint_paths = [local_path(args.stroke_checkpoint), local_path(args.point_checkpoint)]
    configuration = dict(protocol_version=1, manifest=manifest,
                         checkpoints=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in checkpoint_paths],
                         device=str(device), threads=args.threads, point_batch_size=args.point_batch_size,
                         warmup=args.warmup, metric_protocol='grayscale black ink; per-image Otsu; shared evaluation XYW raster; macro image mean')
    if args.binarize:
        configuration['input_preprocessing'] = 'per-image Otsu binary; black ink on white; unchanged 256x256 canvas'
    output.mkdir(parents=True, exist_ok=True)
    config_path = output/'run_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            parser.error('Resume configuration/input/checkpoints differ; use a new output directory')
    else:
        if args.resume and any(output.iterdir()):
            parser.error('Cannot resume output without run_config.json')
        atomic_json(config_path, configuration)
    rows = []
    pending = []
    for index, entry in enumerate(manifest):
        directory = output/'images'/Path(entry['relative']).with_suffix('')
        report_path = directory/'summary.json'
        if report_path.exists():
            record = json.loads(report_path.read_text())
            if record['status'] == 'ok':
                rows.append(record)
                continue
        pending.append((index, entry, directory))
    previous_sessions = json.loads((output/'summary.json').read_text()).get('sessions', []) if (output/'summary.json').exists() else []
    session = dict(processed_images=0, warmup_runs=0, warmup_seconds=0., model_load_seconds=0.)
    sessions = previous_sessions + ([session] if pending else [])
    def snapshot():
        rows.sort(key=lambda r: r['index'])
        report = dict(status='complete' if len(rows) == len(manifest) and all(r['status'] == 'ok' for r in rows) else 'incomplete',
                      selected_images=len(manifest), processed_images=len(rows),
                      metrics=summarize(rows),
                      by_source={name: summarize([r for r in rows if r['source_category'] == name])
                                 for name in sorted({e['source_category'] for e in manifest})},
                      by_script={name: summarize([r for r in rows if r['script_type'] == name])
                                 for name in sorted({e['script_type'] for e in manifest})},
                      sessions=sessions, per_image=rows,
                      timing_protocol='One image at a time; perf_counter; CUDA synchronized; excludes warmup, load, input transfer, CPU metrics, saving and visualization',
                      metric_protocol=configuration['metric_protocol'],
                      input_preprocessing=configuration.get('input_preprocessing', 'original grayscale'),
                      notes='Appearance reconstruction only; no trajectory/order GT. Masks saved as bool >=0.5, not probabilities. '
                            'Macro means over successful images; cap rates are not correctness rates; failed images reported separately.')
        atomic_json(output/'summary.json', report)
        save_csv(output/'per_image.csv', rows)
    if not pending:
        snapshot()
        print('All selected images already complete', flush=True)
        return
    started = perf_counter()
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    stroke, stroke_ck = load_stroke_model(args.stroke_checkpoint, args.device)
    point, point_ck = load_point_model(checkpoint_paths[1], args.device)
    stroke.eval(); point.eval()
    sync()
    session['model_load_seconds'] = perf_counter()-started
    def tensor(image):
        return torch.from_numpy(image.astype(np.float32)[None, None]/255.).to(device)
    @torch.inference_mode()
    def infer(inputs):
        return recover_trajectories(stroke, point, inputs,
                    stroke_threshold=stroke_ck['config']['validity_threshold'],
                    point_threshold=point_ck['config'].get('validity_threshold', .5),
                    point_batch_size=args.point_batch_size)[0]
    print(f'Selected {len(manifest)}, already done {len(rows)}, pending {len(pending)}; warming up {args.warmup} runs', flush=True)
    warm_started = perf_counter()
    for i in range(args.warmup):
        warm_image, _ = prepare_input(read_image(Path(pending[i % len(pending)][1]['image'])), args.binarize)
        inputs = tensor(warm_image)
        warm_result = infer(inputs)
        sync()
        print(f'Warmup {i+1}/{args.warmup}: {len(warm_result["strokes"])} strokes (excluded)', flush=True)
        del warm_result, inputs
    session['warmup_runs'] = args.warmup
    session['warmup_seconds'] = perf_counter()-warm_started
    snapshot()
    last_progress = perf_counter()
    for index, entry, directory in pending:
        directory.mkdir(parents=True, exist_ok=True)
        record = dict(index=index, **entry, prediction_dir=directory.relative_to(output).as_posix())
        try:
            image = read_image(Path(entry['image']))
            model_image, input_threshold = prepare_input(image, args.binarize)
            inputs = tensor(model_image)
            sync()
            inference_started = perf_counter()
            result = infer(inputs)
            sync()
            elapsed = perf_counter()-inference_started
            record.update(status='ok', inference_seconds=elapsed,
                          input_image=str(directory/'input.png'), input_binarized=args.binarize,
                          input_otsu_threshold=input_threshold,
                          **save_prediction(directory, result, model_image, reference_image=image))
            del result, inputs
        except Exception as error:
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            record.update(status='failed', error=str(error), traceback=traceback.format_exc())
            print(f'FAILED {entry["relative"]}: {error}', flush=True)
        atomic_json(directory/'summary.json', record)
        rows.append(record)
        session['processed_images'] += 1
        if len(rows) % 25 == 0 or perf_counter()-last_progress > 15 or len(rows) == len(manifest):
            snapshot()
            successful = [r for r in rows if r['status'] == 'ok']
            mean = float(np.mean([r['inference_seconds'] for r in successful])) if successful else 0.
            print(f'{len(rows)}/{len(manifest)}; mean inference {mean:.4f}s; '
                  f'IoU {np.mean([r["reconstruction"]["iou"] for r in successful]) if successful else 0.:.4f}; '
                  f'ETA {(len(manifest)-len(rows))*(perf_counter()-started)/max(session["processed_images"],1)/60:.1f} min', flush=True)
            last_progress = perf_counter()
    session['wall_seconds_including_load_warmup_and_io'] = perf_counter()-started
    snapshot()
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
