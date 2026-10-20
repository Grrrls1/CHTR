"""Resume cleaned calligraphy masks through a bounded, GPU-assisted single-path route."""
import argparse
from collections import defaultdict
from itertools import islice
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from build_chtr_dataset import ROOT, PIPELINE, finalize_manifest
from continue_chtr_dataset import (checked_output, chunks, cleanup_summary, exclusive_run,
                                   merge_datasets, stage)
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(ROOT / 'CHTR'))
from common import arrays, digest, gray, iter_records, read_json, records, require, write_json, write_records
from assemble_single_path import assemble_group
from models.datasets.lmdb_dataset import load_type_map, validate_sample

TYPE_MAP = ROOT / 'CHTR/reproduction/pipeline_runs/mmah_single_path_gpu2_20261005_five/dataset_simplified_dataloader_v2/stroke_type_map.json'


def validate_training_row(row, names, max_points):
    z = arrays(row['trajectory_path'])
    masks = [gray(s['mask_path']) < 128 for s in sorted(row['stroke_map'], key=lambda s: s['stroke_index'])]
    image = gray(row['image_path'])
    require(image.shape == (256, 256), 'Expected 256 image')
    validate_sample(z['points'], z['stroke_offsets'], z['subpath_offsets'], z['stroke_types'],
                    np.asarray(masks), row, names, 256)
    require(len(z['stroke_types']) <= 64, 'Model max_strokes=64 exceeded')
    require(int(np.diff(z['stroke_offsets']).max()) <= max_points,
            'Model point limit exceeded')


def skeleton_batch(directory, rows, config, references, args, names):
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'dataset.done.json'
    complete = checked_output(marker)
    if complete is not None:
        return complete / 'manifest.jsonl'
    manifest = directory / 'cleanup.jsonl'; report_path = directory / 'cleanup_report.json'
    reports = [read_json(row['cleanup_report_path']) for row in rows]
    for row, report in zip(rows, reports):
        require(row['sample_id'] == report['sample_id'] and row['cleanup_status'] == report['cleanup_status'],
                'Cleanup report mismatch')
    write_records(manifest, rows); write_json(report_path, dict(samples=reports))
    threshold = config['quality_threshold']
    export = stage(directory, 'export', config['pipeline_python'], 'export_clean_masks_for_vs.py',
                   ['--manifest', str(manifest), '--cleanup-report', str(report_path), '--iou', str(threshold)],
                   ['manifest.jsonl', 'rejected.jsonl', 'summary.json'])
    stroke_count = read_json(export / 'summary.json')['selected_strokes']
    if stroke_count:
        paths = stage(directory, 'skeleton', config['pipeline_python'], 'skeleton_gpu_batch.py',
                      ['--manifest', str(export / 'manifest.jsonl'), '--mmah-manifest', str(references),
                       '--device', args.device, '--gpu-batch-size', str(args.gpu_batch_size),
                       '--cpu-workers', str(args.cpu_workers), '--max-points', str(args.max_points),
                       '--iou-threshold', str(threshold)],
                      ['manifest.jsonl', 'errors.jsonl', 'summary.json'], partial_count=stroke_count, resumable=True)
        all_strokes = records(paths / 'manifest.jsonl')
    else:
        all_strokes = []
    passing = [r for r in all_strokes if r['single_path_status'] == 'passed']
    stroke_manifest = directory / 'passing_strokes.jsonl'
    write_records(stroke_manifest, passing)
    groups = defaultdict(list)
    for row in passing:
        groups[row['sample_id']].append(row)
    chars = Path(tempfile.mkdtemp(prefix='characters-', dir=str(directory)))
    packed, rejected = [], []
    for sid, group in sorted(groups.items()):
        try:
            packed.append(assemble_group(group, chars, threshold))
        except ValueError as exc:
            rejected.append(dict(sample_id=sid, error=str(exc)))
    write_records(chars / 'manifest.jsonl', packed)
    write_records(chars / 'rejected.jsonl', rejected)
    dataset = Path(tempfile.mkdtemp(prefix='dataset-', dir=str(directory)))
    finalize_manifest(rows, stroke_manifest, chars / 'manifest.jsonl', report_path,
                      dataset, threshold, config['trajectory_style'], config['copy_assets'])
    # Run the real model contract on every ready sample, including those below whole IoU.
    valid = []; invalid = []
    for row in iter_records(dataset / 'manifest.jsonl'):
        if row['build_status'] == 'trajectory_ready':
            try:
                validate_training_row(row, names, args.max_points)
            except ValueError as exc:
                invalid.append(dict(sample_id=row['sample_id'], error=str(exc)))
                row['build_status'] = 'trajectory_pending'
                row['pending_reason'] = 'model_contract_rejected'
                row['quality']['passed'] = False
                row['rejected_trajectory_path'] = row['trajectory_path']
                row['trajectory_path'] = None
        valid.append(row)
    write_records(dataset / 'manifest.jsonl', valid)
    write_records(dataset / 'model_rejected.jsonl', invalid)
    summary = read_json(dataset / 'summary.json')
    summary.update(ready=sum(r['build_status']=='trajectory_ready' for r in valid),
                   pending=sum(r['build_status']=='trajectory_pending' for r in valid),
                   training_ready=sum(eligible(r) for r in valid), model_rejected=len(invalid),
                   input_strokes=stroke_count, passing_strokes=len(passing),
                   assembled_samples=len(packed), assembly_rejected=len(rejected))
    write_json(dataset / 'summary.json', summary)
    shutil.copyfile(args.stroke_type_map, dataset / 'stroke_type_map.json')
    write_json(marker, dict(output=str(dataset), hashes={name: digest(dataset/name) for name in
               ('manifest.jsonl', 'summary.json', 'model_rejected.jsonl', 'stroke_type_map.json')}))
    print('[batch done] ' + json.dumps(summary, ensure_ascii=False), flush=True)
    return dataset / 'manifest.jsonl'


def eligible(row):
    q = row.get('quality') or {}
    return row.get('build_status') == 'trajectory_ready' and q.get('passed', False) and q.get('all_strokes_passed', False)


def training_selection(dataset, type_map):
    counts = dict(accepted=0, excluded=0)
    def accepted():
        for row in iter_records(dataset / 'manifest.jsonl'):
            if eligible(row):
                counts['accepted'] += 1
                yield row
            else:
                counts['excluded'] += 1
    write_records(dataset / 'training_manifest.jsonl', accepted())
    shutil.copyfile(type_map, dataset / 'stroke_type_map.json')
    write_json(dataset / 'training_summary.json', counts)
    require(counts['accepted'] > 0, 'No model-compatible samples passed QC; inspect batch reports')
    return dataset / 'training_manifest.jsonl'


def pack_training_lmdb(manifest, destination, type_map, map_gb=64, transaction_size=128):
    import lmdb
    from pack_chtr_lmdb import pack_masks
    signature = dict(manifest_sha256=digest(manifest), type_map_sha256=digest(type_map), map_gb=map_gb)
    completion = destination / 'complete.json'
    if completion.exists():
        require(read_json(completion)['signature'] == signature, 'Existing LMDB inputs changed')
        require((destination / 'data.mdb').is_file(), 'Completed LMDB data missing')
        require(digest(destination / 'stroke_type_map.json') == signature['type_map_sha256'],
                'Completed LMDB type map missing or changed')
        return destination
    require(not destination.exists(), 'Incomplete LMDB directory exists; use a new output path')
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='training_lmdb.partial-', dir=str(destination.parent)))
    env = lmdb.open(str(work), map_size=int(map_gb * 1024**3))
    ids, seen = [], set(); txn = env.begin(write=True)
    try:
        for row in iter_records(manifest):
            require(eligible(row), 'Training LMDB received pending/low-quality sample')
            sid = row['sample_id']; require(sid not in seen, 'Duplicate training ID: ' + sid)
            seen.add(sid)
            meta = dict(row, lmdb_key=sid, image_key=sid+'/image', trajectory_key=sid+'/trajectory',
                        masks_key=sid+'/masks', original_image_key=sid+'/original' if row.get('original_image_path') else None)
            for key, pathkey in (('image_key','image_path'), ('trajectory_key','trajectory_path'),
                                 ('original_image_key','original_image_path')):
                if meta[key]:
                    txn.put(meta[key].encode(), Path(row[pathkey]).read_bytes())
            masks = pack_masks(row, Path(manifest).parent)
            require(masks is not None, 'Missing training masks: ' + sid)
            txn.put(meta['masks_key'].encode(), masks)
            txn.put(sid.encode(), json.dumps(meta, ensure_ascii=False, allow_nan=False).encode())
            ids.append(sid)
            if len(ids) % transaction_size == 0:
                txn.commit(); txn = env.begin(write=True)
                print('[lmdb] packed=%d' % len(ids), flush=True)
        require(ids, 'Empty training LMDB')
        txn.put(b'__ids__', json.dumps(ids, ensure_ascii=False).encode())
        txn.put(b'__meta__', json.dumps(dict(schema='chtr-lmdb-v1', count=len(ids), ready=len(ids),
                pending=0, include_masks=True, include_original=True, manifest=str(manifest))).encode())
        txn.commit(); txn = None
        env.sync()
    finally:
        if txn is not None:
            txn.abort()
        env.close()
    shutil.copyfile(type_map, work / 'stroke_type_map.json')
    summary = dict(schema='chtr-lmdb-v1', samples=len(ids), ready=len(ids), pending=0,
                   include_masks=True, include_original=True, source_manifest=str(manifest))
    write_json(work / 'summary.json', summary)
    write_json(work / 'complete.json', dict(signature=signature))
    work.rename(destination)
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--output')
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--gpu-batch-size', type=int, default=64)
    p.add_argument('--cpu-workers', type=int, default=3)
    p.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    p.add_argument('--max-points', type=int, default=1024)
    p.add_argument('--stroke-type-map', default=str(TYPE_MAP))
    p.add_argument('--max-batches', type=int)
    p.add_argument('--pack-lmdb', action='store_true')
    p.add_argument('--lmdb-output', default=str(ROOT / 'CHTR/data/chtr_mmah_lmdb'),
                   help='Final training LMDB directory; separate from intermediate results')
    p.add_argument('--map-size-gb', type=float, default=64)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    require(1 <= args.batch_size <= 500 and 1 <= args.gpu_batch_size <= 256, 'Invalid batch size')
    require(1 <= args.cpu_workers <= 16 and args.max_points >= 2 and args.map_size_gb > 0, 'Invalid resource/point limits')
    require(args.max_batches is None or args.max_batches > 0, 'max-batches must be positive')
    run = Path(args.run).resolve()
    output = Path(args.output).resolve() if args.output else run / 'skeleton_continuation_v1'
    lmdb_output = Path(args.lmdb_output).resolve()
    require(output != run and output not in run.parents and not any(output == run/n or run/n in output.parents
            for n in ('selected','extracted','mask_cleanup','continuation_v1')), 'Use a separate skeleton output directory')
    config = read_json(run / 'run_config.json')
    cleanup = run / 'mask_cleanup/manifest.jsonl'; references = run / 'selected/references.jsonl'
    sources = run / 'selected/all_sources.jsonl'
    summary = cleanup_summary(run / 'mask_cleanup/report.json')
    require(summary['processed'] == summary['input_samples'] and summary['errors'] == 0,
            'Cleanup incomplete; inspect original run')
    require((run / 'mask_cleanup/errors.jsonl').stat().st_size == 0, 'Cleanup has errors')
    _, names = load_type_map(args.stroke_type_map)
    if args.dry_run:
        first = list(islice(iter_records(cleanup), args.batch_size))
        require(first, 'Empty cleanup manifest')
        for row in first:
            require(Path(row['extracted_path']).is_file(), 'Missing cleaned mask')
            require(read_json(row['cleanup_report_path'])['sample_id'] == row['sample_id'], 'Report mismatch')
        print('Validated %d samples; no GPU computation executed.' % len(first))
        print('Plan: cleanup -> export -> %s batch thinning -> CPU graph/QC -> model validation -> merge -> training selection%s' %
              (args.device, ' -> LMDB' if args.pack_lmdb else ''))
        print('Intermediate output: ' + str(output))
        if args.pack_lmdb:
            print('Final LMDB output: ' + str(lmdb_output))
        return
    # Check CUDA before writing any progress state; never fall back silently.
    if args.device == 'cuda':
        import torch
        require(torch.cuda.is_available(), 'Slurm GPU required for --device cuda')
        print('GPU logical cuda:0: ' + torch.cuda.get_device_name(0), flush=True)
    assets = [Path(__file__).resolve(), ROOT/'CHTR/data/continue_chtr_dataset.py',
              ROOT/'CHTR/data/build_chtr_dataset.py', ROOT/'CHTR/data/pack_chtr_lmdb.py',
              ROOT/'CHTR/models/datasets/lmdb_dataset.py', Path(args.stroke_type_map).resolve()]
    assets += [PIPELINE/name for name in ('common.py','export_clean_masks_for_vs.py',
               'skeleton_gpu_batch.py','skeleton_path_trajectory.py','assemble_single_path.py')]
    parameters = {k: getattr(args,k) for k in ('batch_size','gpu_batch_size','cpu_workers','device','max_points')}
    signature = dict(config=config, parameters=parameters,
                     hashes={str(path):digest(path) for path in [cleanup,references,sources]+assets})
    with exclusive_run(output):
        state = output / 'config.json'
        if state.exists():
            require(read_json(state) == signature, 'Inputs/config/code changed; use a new output directory')
        else:
            write_json(state, signature)
        batches = output/'batches'; batches.mkdir(exist_ok=True)
        manifests = []; processed = 0
        for index, rows in enumerate(chunks(iter_records(cleanup), args.batch_size)):
            if args.max_batches is not None and index >= args.max_batches:
                print('[limited run] batches=%d; final merge deferred' % index, flush=True); return
            path = skeleton_batch(batches/('batch_%06d' % index), rows, config, references, args, names)
            manifests.append(path); processed += len(rows)
            print('[progress] batches=%d cleaned_samples=%d/%d' % (index+1,processed,summary['processed']), flush=True)
        require(processed == summary['processed'], 'Cleanup manifest/report count mismatch')
        merge_datasets(manifests,sources,output/'dataset',config['quality_threshold'],config['trajectory_style'],config['copy_assets'])
        training_manifest = training_selection(output/'dataset',args.stroke_type_map)
        if args.pack_lmdb:
            lmdb = pack_training_lmdb(training_manifest,lmdb_output,args.stroke_type_map,args.map_size_gb)
            train_config = read_json(ROOT/'CHTR/models/configs/gpu.json')
            train_config.update(dataset=str(lmdb),output_dir=str(output/'model_training'))
            train_config['model']['max_points'] = args.max_points
            write_json(output/'model_train_config.json',train_config)
        print('[done] ' + str(output),flush=True)


if __name__ == '__main__':
    main()
