"""Build a deterministic single-path 6D trajectory from a cleaned stroke mask.

This is an exploratory baseline for the Virtual Sketching handoff.  It does
not claim that a raster skeleton is the historical pen path; it makes that
limitation explicit in the output metadata while providing a strict one
subpath reference against which multi-path VS output can be compared.
"""
import argparse
import heapq
from pathlib import Path

import cv2
import numpy as np

from common import (SCHEMA, arrays, digest, gray, iou, output_dir, png, preview,
                    records, render, require, states, validate_points,
                    write_json, write_records)


def skeletonize_mask(mask):
    """Return a one-pixel Zhang-Suen skeleton for a binary mask."""
    image = (np.asarray(mask, dtype=np.uint8) != 0).astype(np.uint8)
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        return cv2.ximgproc.thinning(image * 255) > 0
    image = image.copy()
    height, width = image.shape
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            delete = []
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    if image[y, x] == 0:
                        continue
                    p = image[y - 1:y + 2, x - 1:x + 2]
                    n = [int(p[0, 1]), int(p[0, 2]), int(p[1, 2]), int(p[2, 2]),
                         int(p[2, 1]), int(p[2, 0]), int(p[1, 0]), int(p[0, 0])]
                    count = sum(n)
                    transitions = sum(a == 0 and b == 1 for a, b in zip(n, n[1:] + n[:1]))
                    if count < 2 or count > 6 or transitions != 1:
                        continue
                    if phase == 0:
                        keep = n[0] * n[2] * n[4] == 0 and n[2] * n[4] * n[6] == 0
                    else:
                        keep = n[0] * n[2] * n[6] == 0 and n[0] * n[4] * n[6] == 0
                    if keep:
                        delete.append((y, x))
            if delete:
                changed = True
                for y, x in delete:
                    image[y, x] = 0
    return image.astype(bool)


def _graph(skeleton):
    ys, xs = np.where(skeleton)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    index = {tuple(p.astype(int)): i for i, p in enumerate(coords)}
    adjacency = [[] for _ in range(len(coords))]
    for i, (x, y) in enumerate(coords.astype(int)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                j = index.get((x + dx, y + dy))
                if j is not None:
                    adjacency[i].append((j, float(np.hypot(dx, dy))))
    return coords, adjacency


def _dijkstra(adjacency, source):
    distances = np.full(len(adjacency), np.inf, dtype=np.float64)
    previous = np.full(len(adjacency), -1, dtype=np.int64)
    distances[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    return distances, previous


def _path_from_previous(previous, end):
    path = []
    node = int(end)
    while node >= 0:
        path.append(node)
        node = int(previous[node])
    return path[::-1]


def prune_short_branches(skeleton, min_branch_length=4.0):
    """Prune short leaf-to-junction branches, preserving the main component."""
    if min_branch_length <= 0:
        return skeleton.astype(bool)
    image = skeleton.astype(bool).copy()
    while True:
        coords, adjacency = _graph(image)
        if len(coords) < 3:
            return image
        degree = np.asarray([len(items) for items in adjacency])
        leaves = np.where(degree == 1)[0]
        remove = set()
        for leaf in leaves:
            current, previous = int(leaf), -1
            length = 0.0
            branch = [current]
            while True:
                choices = [(n, w) for n, w in adjacency[current] if n != previous]
                if len(choices) != 1 or degree[current] != 2:
                    break
                next_node, weight = choices[0]
                length += weight
                branch.append(next_node)
                previous, current = current, next_node
            if degree[current] >= 3 and length < min_branch_length:
                remove.update(branch[:-1])
        if not remove:
            return image
        for index in remove:
            x, y = coords[index].astype(int)
            image[y, x] = False


def largest_skeleton_component(skeleton):
    count, labels = cv2.connectedComponents(skeleton.astype(np.uint8), 8)
    if count <= 2:
        return skeleton.astype(bool)
    sizes = np.bincount(labels.ravel())
    keep = 1 + int(np.argmax(sizes[1:]))
    return labels == keep


def _median_for_row(row, references):
    source = row.get('mmah_source_npz_path')
    if not source:
        reference = references.get(row.get('character'))
        source = reference.get('source_npz_path') if reference else None
    if not source or not Path(source).is_file():
        return None, 'unavailable'
    with np.load(source, allow_pickle=False) as z:
        key = 'median_%03d' % int(row.get('stroke_index', 0))
        if key not in z:
            return None, 'stroke_median_missing'
        median = np.asarray(z[key], dtype=np.float32)
    require(median.ndim == 2 and median.shape[1] == 2 and len(median) >= 2 and np.isfinite(median).all(),
            'Invalid MMAH median: ' + source)
    return median, 'mmah_ordered_median'


def _nearest_index(coords, point):
    return int(np.argmin(np.linalg.norm(coords - np.asarray(point, dtype=np.float32)[None, :], axis=1)))


def select_main_path(skeleton, median=None):
    """Select the longest connected endpoint path and orient it."""
    coords, adjacency = _graph(skeleton)
    require(len(coords) >= 2, 'Skeleton has fewer than two pixels')
    degree = np.asarray([len(items) for items in adjacency])
    endpoints = np.where(degree == 1)[0].tolist()
    if len(endpoints) < 2:
        seed = endpoints[0] if endpoints else 0
        distances, _ = _dijkstra(adjacency, seed)
        farthest = int(np.nanargmax(distances))
        endpoints = [seed, farthest] if farthest != seed else [seed]
    best = None
    for start in endpoints:
        distances, previous = _dijkstra(adjacency, start)
        for end in endpoints:
            if end == start or not np.isfinite(distances[end]):
                continue
            path = _path_from_previous(previous, end)
            support = 0.0
            if median is not None:
                support = float(np.mean([np.min(np.linalg.norm(coords[path] - p[None, :], axis=1)) <= 10.0
                                         for p in median]))
            # Geodesic length remains primary; median support only breaks ties
            # or selects the visually corresponding branch.
            score = float(distances[end]) + 20.0 * support
            candidate = (score, float(distances[end]), path, support)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    require(best is not None, 'Skeleton endpoints are disconnected')
    path = coords[best[2]].copy()
    direction_status = 'lexicographic_fallback'
    if median is not None:
        if np.linalg.norm(path[-1] - median[0]) < np.linalg.norm(path[0] - median[0]):
            path = path[::-1]
        direction_status = 'mmah_median'
    elif tuple(path[0]) > tuple(path[-1]):
        path = path[::-1]
    return path, dict(endpoint_count=len(endpoints), geodesic_length=best[1],
                      median_support=best[3], direction_status=direction_status)


def _resample_path(path, spacing=1.5, smooth_radius=2):
    path = np.asarray(path, dtype=np.float32)
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    distance = np.r_[0.0, np.cumsum(segment)]
    require(distance[-1] > 0, 'Degenerate centerline')
    count = max(2, int(np.ceil(distance[-1] / spacing)) + 1)
    samples = np.linspace(0.0, distance[-1], count)
    result = np.column_stack([np.interp(samples, distance, path[:, axis]) for axis in range(2)])
    if smooth_radius > 0 and len(result) >= 5:
        radius = int(smooth_radius)
        kernel = np.ones(2 * radius + 1, dtype=np.float32) / float(2 * radius + 1)
        smoothed = np.column_stack([np.convolve(result[:, axis], kernel, mode='same') for axis in range(2)])
        smoothed[:radius] = result[:radius]
        smoothed[-radius:] = result[-radius:]
        result = smoothed.astype(np.float32)
    return result


def _project_to_skeleton(path, skeleton):
    ys, xs = np.where(skeleton)
    support = np.column_stack([xs, ys]).astype(np.float32)
    result = path.copy()
    for i, point in enumerate(result):
        x, y = np.rint(point).astype(int)
        if not (0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1] and skeleton[y, x]):
            result[i] = support[_nearest_index(support, point)]
    return result


def build_trajectory(mask, median=None, spacing=1.5, smooth_radius=2, min_branch_length=4.0):
    mask = np.asarray(mask, dtype=bool)
    require(mask.ndim == 2 and np.any(mask), 'Empty stroke mask')
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    require(components == 2, 'Expected one connected stroke component')
    skeleton = skeletonize_mask(mask)
    skeleton = largest_skeleton_component(skeleton)
    skeleton = prune_short_branches(skeleton, min_branch_length)
    skeleton = largest_skeleton_component(skeleton)
    path, diagnostics = select_main_path(skeleton, median)
    path = _project_to_skeleton(_resample_path(path, spacing, smooth_radius), skeleton)
    distances = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    xy = np.rint(path).astype(int)
    widths = 2.0 * distances[xy[:, 1], xy[:, 0]]
    widths = np.maximum(widths, 1.0).astype(np.float32)
    points = states(np.c_[path.astype(np.float32), widths,
                          np.zeros((len(path), 3), dtype=np.float32)], np.array([0, len(path)], np.int64))
    validate_points(points, np.array([0, len(path)], np.int64), bounds=False)
    diagnostics.update(path_length=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                       point_count=int(len(path)), width_min=float(widths.min()),
                       width_median=float(np.median(widths)), width_max=float(widths.max()),
                       skeleton_pixels=int(skeleton.sum()))
    return points, np.array([0, len(path)], np.int64), skeleton, diagnostics


def _centerline_preview(mask, points, skeleton, path):
    image = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    image[mask] = (215, 215, 215)
    ys, xs = np.where(skeleton)
    image[ys, xs] = (170, 220, 170)
    polyline = np.rint(path[:, :2]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [polyline], False, (0, 100, 220), 1, cv2.LINE_AA)
    for index in np.linspace(0, len(path) - 1, min(8, len(path)), dtype=int):
        x, y = np.rint(path[index, :2]).astype(int)
        cv2.circle(image, (x, y), 2, (0, 0, 180), -1)
    cv2.circle(image, tuple(np.rint(path[0, :2]).astype(int)), 4, (0, 160, 0), -1)
    cv2.circle(image, tuple(np.rint(path[-1, :2]).astype(int)), 4, (0, 0, 255), -1)
    return image


def _overlay(mask, rendered):
    image = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    image[mask] = (0, 0, 255)
    image[rendered >= 128] = (255, 0, 0)
    image[mask & (rendered >= 128)] = (0, 0, 0)
    return image


def process_row(row, destination, references, args):
    mask = gray(row['input_png']) < 128
    median, direction_source = _median_for_row(row, references)
    points, offsets, skeleton, diagnostics = build_trajectory(
        mask, median=median, spacing=args.spacing, smooth_radius=args.smooth_radius,
        min_branch_length=args.min_branch_length)
    rendered = render(points, offsets, size=mask.shape[0], supersample=args.supersample)
    score = iou(rendered >= 128, mask)
    target_area = int(mask.sum())
    render_area = int((rendered >= 128).sum())
    extra_area = int(np.logical_and(rendered >= 128, ~mask).sum())
    path_length = diagnostics['path_length']
    directory = destination / row['stroke_id']
    directory.mkdir()
    trajectory = directory / 'trajectory.npz'
    np.savez_compressed(trajectory, points=points, subpath_offsets=offsets,
                        image_size=np.array(mask.shape[0]),
                        schema=np.array('chtr-single-path-skeleton-v1'))
    render_path = png(directory / 'render.png', 255 - rendered)
    overlay_path = png(directory / 'overlay.png', _overlay(mask, rendered))
    centerline_path = png(directory / 'centerline.png',
                          _centerline_preview(mask, points, skeleton, points[:, :3]))
    status = 'passed' if score >= args.iou_threshold else 'ambiguous'
    item = dict(row, route='skeleton_distance_transform', candidate_id=row.get('stroke_id'),
                trajectory_path=str(trajectory.resolve()), trajectory_sha256=digest(trajectory),
                render_path=render_path, overlay_path=overlay_path, centerline_path=centerline_path,
                qc=dict(iou=score, target_area=target_area, render_area=render_area,
                        extra_area=extra_area, render_area_over_target=(render_area / target_area),
                        path_length=path_length, overdraw_area_ratio=(extra_area / target_area),
                        qc_status=status, threshold=args.iou_threshold),
                single_path_status=status, direction_source=direction_source,
                algorithm=dict(spacing=args.spacing, smooth_radius=args.smooth_radius,
                               min_branch_length=args.min_branch_length, supersample=args.supersample),
                diagnostics=diagnostics, trajectory_review_status='pending')
    write_json(directory / 'metadata.json', item)
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='Single-stroke VS input manifest')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mmah-manifest', default=None,
                        help='Optional ordered MMAH manifest for direction and path support')
    parser.add_argument('--spacing', type=float, default=1.5)
    parser.add_argument('--smooth-radius', type=int, default=2)
    parser.add_argument('--min-branch-length', type=float, default=4.0)
    parser.add_argument('--supersample', type=int, default=4)
    parser.add_argument('--iou-threshold', type=float, default=.87)
    args = parser.parse_args()
    require(args.spacing > 0 and args.smooth_radius >= 0 and args.min_branch_length >= 0,
            'invalid centerline parameters')
    require(1 <= args.supersample <= 8 and 0 < args.iou_threshold <= 1,
            'invalid rendering parameters')
    rows = records(args.manifest)
    unique_ids = [row['stroke_id'] for row in rows]
    require(len(unique_ids) == len(set(unique_ids)), 'Duplicate stroke_id')
    references = {}
    if args.mmah_manifest:
        reference_rows = records(args.mmah_manifest)
        references = {row['character']: row for row in reference_rows}
        require(len(references) == len(reference_rows), 'Duplicate MMAH character')
    out = output_dir(args.output)
    results, errors = [], []
    for row in rows:
        try:
            results.append(process_row(row, out, references, args))
        except Exception as exc:
            errors.append(dict(stroke_id=row.get('stroke_id'), error=str(exc)))
    write_records(out / 'manifest.jsonl', results)
    write_records(out / 'errors.jsonl', errors)
    summary = dict(schema='chtr-single-path-skeleton-v1', route='skeleton_distance_transform',
                   input_manifest=str(Path(args.manifest).resolve()), strokes=len(results),
        errors=len(errors), single_subpath=len(results),
                   passed=sum(row['single_path_status'] == 'passed' for row in results),
                   threshold=args.iou_threshold, exploratory=True,
                   note='Geometric centerline baseline; not a human-approved historical pen path.')
    write_json(out / 'summary.json', summary)
    preview(out, results, 'Single-path skeleton baseline: red=target, blue=render')
    print('Built %d single-path trajectories; %d errors' % (len(results), len(errors)))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
