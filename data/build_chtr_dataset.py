"""Run the CHTR calligraphy pipeline and build a dataloader-friendly dataset.

The default route invokes StrokeExtraction followed by Virtual Sketching.  The
``skeleton`` route is provided as the accepted single-path geometric baseline.
All source records are retained in the final manifest; records without a
finished trajectory are marked ``trajectory_pending`` instead of rejected.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

PIPELINE = Path(__file__).resolve().parents[1] / 'reproduction' / 'pipeline'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE))
from common import arrays, digest, gray, output_dir, read_json, records, require, write_json, write_records


LABEL_NAMES = [
    '点', '横', '竖', '撇', '提', '横折', '横撇/横钩', '横折钩',
    '竖弯钩', '横折弯钩/横斜钩', '竖折折钩', '横折折/横折弯',
    '横折提', '横折折折钩/横撇弯钩', '竖折撇/竖折折', '横折折撇',
    '捺', '斜钩', '弯钩', '竖钩', '竖折/竖弯', '撇折', '撇点', '竖提',
]


def run_stage(name, command, dry_run=False, env=None, allow_partial=False):
    print('[%s] %s' % (name, ' '.join(map(str, command))), flush=True)
    if dry_run:
        return
    started = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, env=env)
    for line in process.stdout:
        print('[%s] %s' % (name, line.rstrip()), flush=True)
    code = process.wait()
    elapsed = time.time() - started
    if allow_partial and code == 1:
        # Only tolerate per-sample failures after the extractor wrote its results.
        stage_output = Path(command[command.index('--output') + 1])
        require((stage_output / 'manifest.jsonl').is_file() and
                (stage_output / 'errors.jsonl').is_file(),
                '%s failed before writing extraction results' % name)
        print('[%s] partial extraction; see errors.jsonl; continuing' % name, flush=True)
    else:
        require(code == 0, '%s failed with exit code %d after %.1fs' % (name, code, elapsed))
    print('[%s] completed in %.1fs' % (name, elapsed), flush=True)


def select_inputs(source_manifest, reference_manifest, output, limit_samples=None, limit_characters=None):
    sources = records(source_manifest)
    references = records(reference_manifest)
    reference_by_character = {row['character']: row for row in references}
    selected = [row for row in sources if row['character'] in reference_by_character]
    if limit_characters:
        chars = []
        for row in selected:
            if row['character'] not in chars:
                chars.append(row['character'])
            if len(chars) >= limit_characters:
                break
        selected = [row for row in selected if row['character'] in set(chars)]
    if limit_samples:
        selected = selected[:limit_samples]
    require(selected, 'No source sample has a matching MMAH reference')
    selected_chars = {row['character'] for row in selected}
    selected_refs = [reference_by_character[c] for c in sorted(selected_chars)]
    write_records(output / 'samples.jsonl', selected)
    write_records(output / 'all_sources.jsonl', sources)
    write_records(output / 'references.jsonl', selected_refs)
    write_json(output / 'selection_summary.json', dict(source_samples=len(sources),
        selected_samples=len(selected), selected_characters=len(selected_chars),
        reference_characters=len(selected_refs), missing_reference_samples=len(sources) - len(selected)))
    print('[select] source=%d selected=%d characters=%d refs=%d' %
          (len(sources), len(selected), len(selected_chars), len(selected_refs)), flush=True)
    limited = bool(limit_samples or limit_characters)
    return selected, (selected if limited else sources)


def copy_file(source, destination):
    source = Path(source)
    if not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def pending_record(source, threshold, trajectory_style):
    sample_id = source['sample_id']
    item = dict(schema='chtr-simplified-manifest-v1', sample_id=sample_id,
                character=source['character'], style='trajectory_style',
                trajectory_style=trajectory_style, original_style=None, author=None,
                manual_review=False, review_status='not_performed',
                human_review_performed=False, build_status='trajectory_pending',
                image_path=source['image_path'], original_image_path=source.get('source_image_path'),
                trajectory_path=None, stroke_count=int(source.get('evaluation_stroke_count', 0)),
                quality=dict(score_name='stage_iou.trajectory_whole', score=None,
                             threshold=threshold, passed=False, filter_in_dataloader=True),
                stage_iou={}, stroke_types=[], stroke_type_names=[], stroke_map=[],
                provenance=dict(source=source.get('source'),
                                provenance_status=source.get('provenance_status'),
                                source_image_path=source.get('source_image_path')),
                pending_reason='missing_reference_or_trajectory',
                quality_notes='Source retained; trajectory is pending.')
    return item


def finalize_manifest(selected, stroke_manifest, character_manifest, cleanup_report,
                      output, threshold, trajectory_style, copy_assets):
    stroke_rows = defaultdict(list)
    for row in records(stroke_manifest):
        stroke_rows[row['sample_id']].append(row)
    character_rows = {row['sample_id']: row for row in records(character_manifest)}
    cleanup_rows = {row['sample_id']: row for row in read_json(cleanup_report)['samples']}
    result = []
    counts = defaultdict(int)
    for index, source in enumerate(selected, 1):
        sample_id = source['sample_id']
        rows = sorted(stroke_rows.get(sample_id, []), key=lambda row: row['stroke_index'])
        character = character_rows.get(sample_id)
        destination = output / 'samples' / sample_id
        if character is None or not rows:
            item = pending_record(source, threshold, trajectory_style)
            result.append(item); counts['pending'] += 1
            continue
        source_package = Path(source.get('source_image_path', source['image_path']))
        target_image = Path(source['image_path'])
        char_npz = Path(character['npz_path'])
        destination.mkdir(parents=True, exist_ok=True)
        if copy_assets:
            image_path = copy_file(target_image, destination / 'image_256.png')
            original_path = copy_file(source_package, destination / ('original_calligraphy' + source_package.suffix))
            trajectory_path = copy_file(char_npz, destination / 'trajectory.npz')
        else:
            image_path, original_path, trajectory_path = str(target_image), str(source_package), str(char_npz)
        cleanup = cleanup_rows.get(sample_id, {})
        per_iou = [float(row['qc']['iou']) for row in rows]
        with np.load(str(char_npz), allow_pickle=False) as z:
            labels = z['stroke_types'].astype(np.int64).tolist()
            stroke_offsets = z['stroke_offsets'].astype(np.int64).tolist()
        stroke_map = []
        for stroke_index, row in enumerate(rows):
            mask_path = row.get('input_png')
            if copy_assets:
                mask_path = copy_file(mask_path, destination / 'stroke_masks' / ('stroke%03d.png' % stroke_index))
            stroke_map.append(dict(stroke_index=stroke_index, stroke_id=row['stroke_id'],
                                   stroke_label=int(row['stroke_label']),
                                   stroke_type_name=LABEL_NAMES[int(row['stroke_label'])],
                                   point_range=[int(stroke_offsets[stroke_index]), int(stroke_offsets[stroke_index + 1])],
                                   stroke_iou=float(row['qc']['iou']),
                                   stroke_quality_passed=float(row['qc']['iou']) >= threshold,
                                   mask_path=mask_path))
        whole_iou = float(character['whole_iou'])
        item = dict(schema='chtr-simplified-manifest-v1', sample_id=sample_id,
                    character=source['character'], style='trajectory_style',
                    trajectory_style=trajectory_style, original_style=None, author=None,
                    manual_review=False, review_status='not_performed',
                    human_review_performed=False, build_status='trajectory_ready',
                    image_path=image_path, original_image_path=original_path,
                    trajectory_path=trajectory_path,
                    image_sha256=digest(image_path), original_image_sha256=digest(original_path),
                    trajectory_sha256=digest(trajectory_path), stroke_count=len(rows),
                    stroke_types=labels, stroke_type_names=[LABEL_NAMES[int(v)] for v in labels],
                    stage_iou=dict(mask_extraction_whole=float(cleanup.get('before', {}).get('iou', np.nan)),
                                   mask_cleanup_whole=float(cleanup.get('after', {}).get('iou', np.nan)),
                                   trajectory_stroke_min=min(per_iou), trajectory_stroke_mean=float(np.mean(per_iou)),
                                   trajectory_stroke_median=float(np.median(per_iou)), trajectory_stroke_max=max(per_iou),
                                   trajectory_whole=whole_iou, upstream_fidelity=None, native_replay=None),
                    quality=dict(score_name='stage_iou.trajectory_whole', score=whole_iou,
                                 threshold=threshold, passed=whole_iou >= threshold,
                                 all_strokes_passed=min(per_iou) >= threshold, filter_in_dataloader=True),
                    stroke_map=stroke_map,
                    provenance=dict(source=source.get('source'), provenance_status=source.get('provenance_status'),
                                    source_image_path=source.get('source_image_path')),
                    quality_notes='All samples retained; dataloader-side IoU filtering expected.')
        result.append(item); counts['ready'] += 1
        if index == 1 or index % 100 == 0 or index == len(selected):
            print('[finalize] %d/%d ready=%d pending=%d' %
                  (index, len(selected), counts['ready'], counts['pending']), flush=True)
    write_records(output / 'manifest.jsonl', result)
    write_json(output / 'summary.json', dict(schema='chtr-simplified-manifest-v1', samples=len(result),
        ready=counts['ready'], pending=counts['pending'], threshold=threshold,
        style='trajectory_style', trajectory_style=trajectory_style,
        note='All source samples retained; dataloader applies quality filters.'))
    print('[finalize] wrote %d records' % len(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', default=str(ROOT / 'datasets/processed/chinese-calligraphy-221030/manifest.jsonl'))
    parser.add_argument('--mmah-bank', default=str(ROOT / 'CHTR/reproduction/pipeline_runs/mmah_reference_bank_20260918/manifest.jsonl'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--route', choices=('vs', 'skeleton'), default='vs')
    parser.add_argument('--limit-samples', type=int)
    parser.add_argument('--limit-characters', type=int)
    parser.add_argument('--quality-threshold', type=float, default=.87)
    parser.add_argument('--trajectory-style', default='MMAH宋体')
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--vs-gpu', default=None,
                        help='Physical GPU for Virtual Sketching; defaults to --gpu')
    parser.add_argument('--vs-device', choices=('cpu', 'cuda'), default='cpu',
                        help='Virtual Sketching device; default preserves historical CPU inference')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--copy-assets', action='store_true')
    parser.add_argument('--allow-extraction-errors', action='store_true',
                        help='Continue with successful extractions; retain failed sources as pending')
    parser.add_argument('--resume-extraction', action='store_true',
                        help='Reuse completed extractions in a run that has not started cleanup')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--pipeline-python', default=str(ROOT / 'CHTR/third_party/StrokeExtraction/.venv/bin/python'),
                        help='Python for non-TF pipeline stages and the driver environment')
    parser.add_argument('--stroke-python', default=str(ROOT / 'CHTR/third_party/StrokeExtraction/.venv/bin/python'))
    parser.add_argument('--vs-python', default=str(ROOT / '.venvs/virtual-sketching/bin/python'))
    parser.add_argument('--vs-repo', default=str(ROOT / 'CHTR/third_party/virtual_sketching'))
    parser.add_argument('--stroke-repo', default=str(ROOT / 'CHTR/third_party/StrokeExtraction'))
    parser.add_argument('--vs-model', default='pretrain_clean_line_drawings')
    parser.add_argument('--vs-samples', type=int, default=3)
    parser.add_argument('--vs-seed', type=int, default=0)
    parser.add_argument('--start-cursor-source', choices=('random', 'skeleton', 'mmah'), default='random')
    parser.add_argument('--sdnet-model', default=str(ROOT / 'CHTR/third_party/StrokeExtraction/out/SDNet_MMAH_CCSEDB_v4/model/sdnet_model.pth'))
    parser.add_argument('--segnet-model', default=str(ROOT / 'CHTR/third_party/StrokeExtraction/out/SegNet_MMAH_CCSEDB_v4/model/model.pth'))
    parser.add_argument('--extractnet-model', default=str(ROOT / 'CHTR/third_party/StrokeExtraction/out/ExtractNet_MMAH_CCSEDB_v4/model/model_extract.pth'))
    args = parser.parse_args()
    if args.vs_gpu is None:
        args.vs_gpu = args.gpu
    require(0 < args.quality_threshold <= 1, 'invalid quality threshold')
    out = Path(args.output).resolve() if args.resume_extraction else output_dir(args.output)
    if args.resume_extraction:
        require(out.is_dir() and (out / 'selected').is_dir(), 'Resume run not found')
        require(not any((out / name).exists() for name in
                        ('mask_cleanup', 'vs_inputs', 'single_path', 'vs_raw', 'vs_qc', 'characters', 'dataset')),
                'resume-extraction only supports runs before cleanup')
    selected_dir = out / 'selected'
    if args.resume_extraction:
        with tempfile.TemporaryDirectory(prefix='resume-check-', dir=str(out)) as check:
            selected, manifest_sources = select_inputs(args.source_manifest, args.mmah_bank, Path(check),
                                                       args.limit_samples, args.limit_characters)
            for name in ('samples.jsonl', 'all_sources.jsonl', 'references.jsonl'):
                require(digest(Path(check) / name) == digest(selected_dir / name),
                        'Resume selection changed: ' + name)
    else:
        selected_dir.mkdir()
        selected, manifest_sources = select_inputs(args.source_manifest, args.mmah_bank, selected_dir,
                                                   args.limit_samples, args.limit_characters)
    write_json(out / 'run_config.json', vars(args))
    extracted = out / 'extracted'
    cleanup = out / 'mask_cleanup'
    vs_inputs = out / 'vs_inputs'
    run_stage('extract', [args.stroke_python, str(PIPELINE / 'extract_strokes_batch.py'),
        '--manifest', str(selected_dir / 'samples.jsonl'), '--references', str(selected_dir / 'references.jsonl'),
        '--output', str(extracted), '--repo', args.stroke_repo, '--gpu', args.gpu, '--device', args.device,
        '--sdnet-model', args.sdnet_model, '--segnet-model', args.segnet_model,
        '--extractnet-model', args.extractnet_model] +
        (['--resume'] if args.resume_extraction else []), args.dry_run,
        allow_partial=args.allow_extraction_errors)
    run_stage('cleanup', [args.pipeline_python, str(PIPELINE / 'clean_stroke_masks.py'),
        '--manifest', str(extracted / 'manifest.jsonl'), '--output', str(cleanup), '--whole-iou', str(args.quality_threshold)], args.dry_run)
    run_stage('export', [args.pipeline_python, str(PIPELINE / 'export_clean_masks_for_vs.py'),
        '--manifest', str(cleanup / 'manifest.jsonl'), '--cleanup-report', str(cleanup / 'report.json'),
        '--output', str(vs_inputs), '--iou', str(args.quality_threshold)], args.dry_run)
    if args.route == 'skeleton':
        strokes = out / 'single_path'
        chars = out / 'characters'
        run_stage('single-path', [args.pipeline_python, str(PIPELINE / 'skeleton_path_trajectory.py'),
            '--manifest', str(vs_inputs / 'manifest.jsonl'), '--mmah-manifest', str(selected_dir / 'references.jsonl'),
            '--output', str(strokes), '--iou-threshold', str(args.quality_threshold)], args.dry_run)
        run_stage('assemble', [args.pipeline_python, str(PIPELINE / 'assemble_single_path.py'),
            '--manifest', str(strokes / 'manifest.jsonl'), '--output', str(chars),
            '--iou-threshold', str(args.quality_threshold)], args.dry_run)
        stroke_manifest, character_manifest = strokes / 'manifest.jsonl', chars / 'manifest.jsonl'
    else:
        raw = out / 'vs_raw'
        qc = out / 'vs_qc'
        chars = out / 'characters'
        vs_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.vs_gpu))
        run_stage('vs', [args.vs_python, str(PIPELINE / 'vectorize_strokes_batch.py'),
            '--manifest', str(vs_inputs / 'manifest.jsonl'), '--output', str(raw), '--repo', args.vs_repo,
            '--model', args.vs_model, '--samples', str(args.vs_samples), '--seed', str(args.vs_seed),
            '--start-cursor-source', args.start_cursor_source,
            '--vs-device', args.vs_device,
            '--mmah-manifest', str(selected_dir / 'references.jsonl')], args.dry_run, env=vs_env)
        run_stage('render-qc', [args.pipeline_python, str(PIPELINE / 'render_and_filter.py'),
            '--manifest', str(raw / 'manifest.jsonl'), '--output', str(qc),
            '--iou', str(args.quality_threshold), '--fidelity', str(args.quality_threshold)], args.dry_run)
        run_stage('assemble', [args.pipeline_python, str(PIPELINE / 'assemble_vs_exploratory.py'),
            '--manifest', str(qc / 'best.jsonl'), '--output', str(chars),
            '--iou', str(args.quality_threshold)], args.dry_run)
        stroke_manifest, character_manifest = qc / 'best.jsonl', chars / 'manifest.jsonl'
    if not args.dry_run:
        finalize_manifest(manifest_sources, stroke_manifest, character_manifest, cleanup / 'report.json',
                          out / 'dataset', args.quality_threshold, args.trajectory_style, args.copy_assets)
    write_json(out / 'run_config.json', vars(args))
    print('[done] output=%s route=%s' % (out, args.route), flush=True)


if __name__ == '__main__':
    main()
