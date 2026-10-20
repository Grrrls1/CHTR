"""Resident three-stage extractor with bounded background saving and safe resume."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
from pathlib import Path
import sys
import time
import numpy as np
from common import ROOT, records, unique, output_dir, write_records, arrays, gray, masks_and_labels, write_json, read_json, digest, require


def save_result(destination, result, item):
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / 'all_stages.partial.npz'
    np.savez_compressed(str(temporary), **result)
    os.replace(str(temporary), str(destination / 'all_stages.npz'))
    # Metadata is the completion marker, written only after a complete NPZ.
    write_json(destination / 'metadata.json', item)
    return item


def cached_result(destination, row, model_hashes, reference_hash, device):
    metadata = destination / 'metadata.json'
    if not metadata.exists():
        return None
    old = read_json(metadata)
    require(old['model_sha256'] == model_hashes, 'Resume model changed')
    require(old['reference_sha256'] == reference_hash, 'Resume reference changed')
    require(old['inference_device'] == device, 'Resume device changed')
    require(all(old.get(k) == v for k, v in row.items()), 'Resume source metadata changed')
    if 'input_sha256' in old:
        require(old['input_sha256'] == digest(row['image_path']), 'Resume image changed')
    else:
        # Legacy artifacts have no normalized-image hash. Only reuse unchanged
        # files older than their completion marker, with identical source metadata.
        require(Path(row['image_path']).stat().st_mtime_ns <= metadata.stat().st_mtime_ns,
                'Legacy resume image modified after extraction')
    require(Path(old['extracted_path']).resolve() == (destination / 'all_stages.npz').resolve(),
            'Resume NPZ path changed')
    z = arrays(old['extracted_path'])
    masks_and_labels(z['extractnet_strokes'], z['stroke_labels'])
    return old


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--references', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--repo', default=str(ROOT / 'CHTR/third_party/StrokeExtraction'))
    p.add_argument('--gpu', default='0', help='Physical CUDA device exposed as logical cuda:0')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--sdnet-model'); p.add_argument('--segnet-model'); p.add_argument('--extractnet-model')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--save-workers', type=int, default=2)
    a = p.parse_args()
    require(1 <= a.save_workers <= 8, 'save-workers must be 1..8')
    rows = records(a.manifest); unique(rows)
    refs = records(a.references)
    require(len({r['character'] for r in refs}) == len(refs), 'Duplicate reference character')
    refs = {r['character']: r for r in refs}
    out = Path(a.output).resolve() if a.resume else output_dir(a.output)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(a.repo).resolve()
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    sys.path.insert(0, str(repo)); os.chdir(str(repo))
    import torch
    if a.device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA unavailable')
    from extraction_stroke_application_for_single_character_ import ExtractStroke
    checkpoints = [a.sdnet_model or repo/'out/SDNet_RHSEDB/model/sdnet_model.pth',
                   a.segnet_model or repo/'out/SegNet_RHSEDB/model/model.pth',
                   a.extractnet_model or repo/'out/ExtractNet_RHSEDB/model/model_extract.pth']
    model = ExtractStroke(*[str(Path(v).resolve()) for v in checkpoints],
                         inference_device=torch.device('cuda:0' if a.device == 'cuda' else 'cpu'))
    model_hashes = [digest(v) for v in checkpoints]

    @lru_cache(maxsize=16)
    def reference_data(character):
        ref = refs[character]
        z = arrays(ref['reference_path'])
        masks, labels = z['reference_single_image'], z['stroke_label']
        masks_and_labels(masks, labels)
        color, centers = model.get_reference_data(masks, labels)
        return masks, labels, color, centers, digest(ref['reference_path'])

    success, errors, pending = [], [], deque()
    started = time.monotonic()
    computed = reused = 0
    recent_times = deque(maxlen=200)
    with ThreadPoolExecutor(max_workers=a.save_workers) as writers:
        for index, row in enumerate(rows, 1):
            iteration_started = time.monotonic()
            # Saving errors are fatal (e.g. disk full), not sample failures.
            while pending and pending[0].done():
                success.append(pending.popleft().result())
            if len(pending) >= 2 * a.save_workers:
                success.append(pending.popleft().result())
            masks, labels, color, centers, reference_hash = reference_data(row['character'])
            destination = out / row['sample_id']
            old = cached_result(destination, row, model_hashes, reference_hash, a.device) if a.resume else None
            if old is not None:
                success.append(old); reused += 1
            else:
                try:
                    image = gray(row['image_path'])
                    require(image.shape == (256, 256), 'Source not normalized')
                    result = model.get_extract_strokes(dict(
                        target_image=(1-image.astype(np.float32)/255)[None],
                        reference_single_image=masks, stroke_label=labels,
                        reference_color_image=color, reference_single_centroid=centers))
                    masks_and_labels(result['extractnet_strokes'], labels)
                    item = dict(row, extracted_path=str(destination/'all_stages.npz'),
                                review_status='pending', expected_strokes=len(labels),
                                model_sha256=model_hashes, reference_sha256=reference_hash,
                                input_sha256=digest(row['image_path']),
                                category_system='StrokeExtraction-24', inference_device=a.device)
                    pending.append(writers.submit(save_result, destination, result, item))
                except Exception as exc:
                    errors.append(dict(sample_id=row['sample_id'], error=str(exc)))
                    print('Failed %s: %s' % (row['sample_id'], exc), flush=True)
                computed += 1
                recent_times.append(time.monotonic() - iteration_started)
            elapsed = time.monotonic() - started
            seconds_per_sample = sum(recent_times) / len(recent_times) if recent_times else 0
            if index % 10 == 0 or index == len(rows) or index == 1:
                print('Progress %d/%d (%.2f%%) saved=%d saving=%d failed=%d reused=%d elapsed=%.0fs ETA=%.0fs speed=%.2f/s' %
                      (index, len(rows), 100*index/len(rows), len(success), len(pending), len(errors),
                       reused, elapsed, seconds_per_sample*(len(rows)-index),
                       1/seconds_per_sample if seconds_per_sample else 0), flush=True)
        while pending:
            success.append(pending.popleft().result())
    write_records(out/'manifest.jsonl', success)
    write_records(out/'errors.jsonl', errors)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
