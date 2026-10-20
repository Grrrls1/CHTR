"""Batched Zhang-Suen thinning on CUDA, with CPU graph tracing and canonical QC."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
from pathlib import Path
import time

import cv2
import numpy as np

from common import (arrays, digest, gray, iou, iter_records, output_dir, png, preview,
                    read_json, records, require, states, validate_points, write_json, write_records)
import skeleton_path_trajectory as baseline


def thin_batch(masks, device='cuda'):
    """Same simultaneous-deletion rules as the baseline's Zhang-Suen fallback."""
    import torch
    image = torch.as_tensor(np.asarray(masks, dtype=bool), device=device).clone()
    require(image.ndim == 3 and min(image.shape[1:]) >= 3, 'Expected [N,H,W] masks')
    # Every non-final iteration must remove at least one pixel from the batch.
    with torch.inference_mode():
        for _ in range(image.numel() + 1):
            changed = torch.zeros((), dtype=torch.bool, device=device)
            for phase in (0, 1):
                n = (image[:, :-2, 1:-1], image[:, :-2, 2:], image[:, 1:-1, 2:],
                     image[:, 2:, 2:], image[:, 2:, 1:-1], image[:, 2:, :-2],
                     image[:, 1:-1, :-2], image[:, :-2, :-2])
                count = torch.zeros_like(n[0], dtype=torch.uint8)
                transitions = torch.zeros_like(count)
                for i in range(8):
                    count += n[i].to(torch.uint8)
                    transitions += (~n[i] & n[(i + 1) % 8]).to(torch.uint8)
                if phase == 0:
                    keep = ~(n[0] & n[2] & n[4]) & ~(n[2] & n[4] & n[6])
                else:
                    keep = ~(n[0] & n[2] & n[6]) & ~(n[0] & n[4] & n[6])
                center = image[:, 1:-1, 1:-1]
                delete = center & (count >= 2) & (count <= 6) & (transitions == 1) & keep
                changed |= delete.any()
                center &= ~delete
            if not changed.item():
                return image.cpu().numpy()
    raise ValueError('Thinning did not converge')


def trajectory_from_skeleton(mask, skeleton, median, settings):
    require(cv2.connectedComponents(mask.astype(np.uint8), 8)[0] == 2,
            'Expected one connected stroke component')
    skeleton = baseline.largest_skeleton_component(skeleton)
    skeleton = baseline.prune_short_branches(skeleton, settings['min_branch_length'])
    skeleton = baseline.largest_skeleton_component(skeleton)
    path, diagnostics = baseline.select_main_path(skeleton, median)
    path = baseline._resample_path(path, settings['spacing'], settings['smooth_radius'])
    before = len(path)
    if before > settings['max_points']:
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        locations = np.linspace(0., distance[-1], settings['max_points'])
        path = np.column_stack([np.interp(locations, distance, path[:, i]) for i in range(2)])
    path = baseline._project_to_skeleton(path, skeleton)
    distances = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    xy = np.rint(path).astype(int)
    widths = np.maximum(2. * distances[xy[:, 1], xy[:, 0]], 1.).astype(np.float32)
    offsets = np.array([0, len(path)], np.int64)
    points = states(np.c_[path.astype(np.float32), widths, np.zeros((len(path), 3), np.float32)], offsets)
    validate_points(points, offsets, size=mask.shape[0], bounds=True)
    require(len(points) <= settings['max_points'], 'Point limit exceeded')
    diagnostics.update(point_count=len(points), points_before_limit=before,
                       resampled_for_limit=before > len(points), skeleton_pixels=int(skeleton.sum()),
                       path_length=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()))
    return points, offsets, skeleton, diagnostics


def process_stroke(row, mask, skeleton, median, direction, settings, signature, output):
    cv2.setNumThreads(1)
    points, offsets, skeleton, diagnostics = trajectory_from_skeleton(mask, skeleton, median, settings)
    rendered = baseline.render(points, offsets, size=mask.shape[0], supersample=settings['supersample'])
    score = iou(rendered >= 128, mask)
    directory = Path(output) / row['stroke_id']; directory.mkdir(parents=True, exist_ok=True)
    trajectory = directory / 'trajectory.npz'
    temporary = directory / 'trajectory.partial.npz'
    np.savez_compressed(temporary, points=points, subpath_offsets=offsets,
                        image_size=np.array(mask.shape[0]), schema=np.array('chtr-single-path-skeleton-v1'))
    temporary.replace(trajectory)
    status = 'passed' if score >= settings['iou_threshold'] else 'ambiguous'
    item = dict(row, route='skeleton_distance_transform', candidate_id=row['stroke_id'],
                trajectory_path=str(trajectory.resolve()), trajectory_sha256=digest(trajectory),
                render_path=png(directory / 'render.png', 255 - rendered),
                overlay_path=png(directory / 'overlay.png', baseline._overlay(mask, rendered)),
                centerline_path=png(directory / 'centerline.png', baseline._centerline_preview(mask, points, skeleton, points[:, :3])),
                qc=dict(iou=score, qc_status=status, threshold=settings['iou_threshold']),
                single_path_status=status, direction_source=direction, diagnostics=diagnostics,
                algorithm=dict(settings), signature=signature, trajectory_review_status='pending')
    write_json(directory / 'metadata.json', item)
    return item


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    p.add_argument('--mmah-manifest', required=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--gpu-batch-size', type=int, default=64)
    p.add_argument('--cpu-workers', type=int, default=3)
    p.add_argument('--max-points', type=int, default=1024)
    p.add_argument('--iou-threshold', type=float, default=.87)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    require(1 <= a.gpu_batch_size <= 256 and 1 <= a.cpu_workers <= 16, 'Invalid batch/workers')
    require(a.max_points >= 2 and 0 < a.iou_threshold <= 1, 'Invalid trajectory settings')
    import torch
    if a.device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA requested but unavailable; no automatic CPU fallback')
        print('GPU logical cuda:0: ' + torch.cuda.get_device_name(0), flush=True)
    torch.set_num_threads(1); cv2.setNumThreads(1)
    rows = records(a.manifest)
    require(len({r['stroke_id'] for r in rows}) == len(rows), 'Duplicate stroke ID')
    out = Path(a.output).resolve() if a.resume else output_dir(a.output)
    out.mkdir(parents=True, exist_ok=True)
    refs = {r['character']: r for r in iter_records(a.mmah_manifest)}
    settings = dict(spacing=1.5, smooth_radius=2, min_branch_length=4., supersample=4,
                    max_points=a.max_points, iou_threshold=a.iou_threshold,
                    thinning='zhang_suen_torch', device=a.device)
    code_hash = {name: digest(Path(__file__).parent / name)
                 for name in ('skeleton_gpu_batch.py', 'skeleton_path_trajectory.py', 'common.py')}
    references_hash = digest(a.mmah_manifest)
    results, errors, pending = {}, {}, {}
    started = time.monotonic(); reused = 0; thin_seconds = 0.

    def collect(done):
        for future in done:
            sid = pending.pop(future)
            try:
                results[sid] = future.result()
            except Exception as exc:
                errors[sid] = dict(stroke_id=sid, error=str(exc))
                print('Failed %s: %s' % (sid, exc), flush=True)
        completed = len(results) + len(errors)
        print('Progress %d/%d saved=%d failed=%d reused=%d elapsed=%.1fs' %
              (completed, len(rows), len(results), len(errors), reused, time.monotonic()-started), flush=True)

    with ProcessPoolExecutor(max_workers=a.cpu_workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        remaining = []
        for row in rows:
            signature = dict(input_sha256=digest(row['input_png']), references_sha256=references_hash,
                             code_sha256=code_hash, settings=settings)
            cached = out / row['stroke_id'] / 'metadata.json'
            if a.resume and cached.exists():
                old = read_json(cached)
                require(old['signature'] == signature, 'Cached stroke inputs/config changed: ' + row['stroke_id'])
                require(digest(old['trajectory_path']) == old['trajectory_sha256'], 'Cached trajectory changed')
                require(all(Path(old[k]).is_file() for k in ('render_path', 'overlay_path', 'centerline_path')),
                        'Cached preview missing')
                results[row['stroke_id']] = old; reused += 1
            else:
                remaining.append((row, signature))
        for start in range(0, len(remaining), a.gpu_batch_size):
            group = remaining[start:start+a.gpu_batch_size]
            masks = [gray(row['input_png']) < 128 for row, _ in group]
            require(all(mask.shape == (256, 256) for mask in masks), 'Expected 256 masks')
            t = time.monotonic()
            skeletons = thin_batch(masks, a.device)
            thin_seconds += time.monotonic()-t
            if a.device == 'cuda' and start == 0:
                check_count = min(3, len(masks))
                expected = thin_batch(masks[:check_count], 'cpu')
                require(np.array_equal(skeletons[:check_count], expected),
                        'CUDA thinning differs from CPU reference; stop before saving trajectories')
                print('CUDA/CPU thinning parity passed: %d masks' % check_count, flush=True)
            for (row, signature), mask, skeleton in zip(group, masks, skeletons):
                if len(pending) >= 2*a.cpu_workers:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED); collect(done)
                median, direction = baseline._median_for_row(row, refs)
                future = pool.submit(process_stroke, row, mask, skeleton, median, direction,
                                     settings, signature, str(out))
                pending[future] = row['stroke_id']
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED); collect(done)
    ordered = [results[r['stroke_id']] for r in rows if r['stroke_id'] in results]
    write_records(out / 'manifest.jsonl', ordered)
    write_records(out / 'errors.jsonl', [errors[r['stroke_id']] for r in rows if r['stroke_id'] in errors])
    write_json(out / 'summary.json', dict(strokes=len(ordered), errors=len(errors), reused=reused,
               passed=sum(r['single_path_status']=='passed' for r in ordered), device=a.device,
               thinning_seconds=thin_seconds, elapsed_seconds=time.monotonic()-started,
               gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else 0,
               exploratory=True, settings=settings))
    preview(out, ordered, 'Single-path skeleton: canonical CPU raster QC', limit=200)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
