"""Resume a completed cleanup in bounded, independently checkpointed VS batches."""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
from itertools import islice
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

from build_chtr_dataset import PIPELINE, ROOT, finalize_manifest, pending_record, copy_file
sys.path.insert(0, str(PIPELINE))
from common import digest, iter_records, read_json, require, write_json, write_records


def chunks(rows, size):
    rows = iter(rows)
    while True:
        batch = list(islice(rows, size))
        if not batch:
            return
        yield batch


def cleanup_summary(path):
    """Read only the small summary header, leaving the huge samples array on disk."""
    header = []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            require(sum(map(len, header)) + len(line) < 65536, 'Cleanup header exceeds 64 KiB')
            if line.strip() == '"samples": [':
                return json.loads(''.join(header) + '"samples": []}')
            header.append(line)
    raise ValueError('Cleanup report has no samples array')


@contextmanager
def exclusive_run(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another continuation is already running in ' + str(directory))
        yield


def checked_output(marker):
    if not marker.exists():
        return None
    state = read_json(marker)
    for name, expected in state['hashes'].items():
        require(digest(Path(state['output']) / name) == expected,
                'Completed stage manifest changed: ' + str(marker))
    return Path(state['output'])


def stage(batch_dir, name, python, script, arguments, outputs, partial_count=None, resumable=False):
    marker = batch_dir / (name + '.done.json')
    old = checked_output(marker)
    if old is not None:
        print('[%s] reused %s' % (name, old), flush=True)
        return old
    work_marker = batch_dir / (name + '.work.json')
    if resumable and work_marker.exists():
        output = Path(read_json(work_marker)['output'])
        extra = ['--resume']
    else:
        output = Path(tempfile.mkdtemp(prefix=name + '-', dir=str(batch_dir)))
        write_json(work_marker, dict(output=str(output)))
        extra = []
    command = [python, str(PIPELINE / script)] + arguments + ['--output', str(output)] + extra
    print('[%s] %s' % (name, ' '.join(command)), flush=True)
    with (batch_dir / (name + '.log')).open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        with process.stdout:
            for line in process.stdout:
                log.write(line); log.flush()
                print('[%s] %s' % (name, line.rstrip()), flush=True)
        code = process.wait()
    require(code in (0, 1) if partial_count is not None else code == 0,
            '%s failed (exit %s); log=%s' % (name, code, batch_dir / (name + '.log')))
    for filename in outputs:
        require((output / filename).is_file(), name + ' missing ' + filename)
    errors = 0
    if partial_count is not None:
        errors = sum(1 for _ in iter_records(output / 'errors.jsonl'))
        successful = sum(1 for _ in iter_records(output / 'manifest.jsonl'))
        require(successful + errors == partial_count, name + ' wrote incomplete candidate accounting')
        require(code == 0 or errors > 0, name + ' exit 1 without recorded candidate errors')
    write_json(marker, dict(output=str(output), errors=errors,
                           hashes={f: digest(output / f) for f in outputs}))
    if errors:
        print('[%s] recorded %d candidate errors; continuing with available results' % (name, errors), flush=True)
    return output


def run_batch(directory, rows, config, threshold, references, copy_assets):
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'dataset.done.json'
    ready = checked_output(marker)
    if ready is not None:
        return ready / 'manifest.jsonl'
    cleanup_manifest = directory / 'cleanup.jsonl'
    cleanup_report = directory / 'cleanup_report.json'
    write_records(cleanup_manifest, rows)
    # Read only this batch's per-image reports; never load the 1.2 GB report.json.
    reports = [read_json(row['cleanup_report_path']) for row in rows]
    for row, report in zip(rows, reports):
        require(row['sample_id'] == report['sample_id'], 'Cleanup report sample mismatch')
        require(row['cleanup_status'] == report['cleanup_status'], 'Cleanup status mismatch')
    write_json(cleanup_report, dict(samples=reports))
    python = config['pipeline_python']
    export = stage(directory, 'export', python, 'export_clean_masks_for_vs.py',
                   ['--manifest', str(cleanup_manifest), '--cleanup-report', str(cleanup_report),
                    '--iou', str(threshold)], ['manifest.jsonl', 'rejected.jsonl', 'summary.json'])
    exported_count = read_json(export / 'summary.json')['selected_strokes']
    if exported_count:
        raw = stage(directory, 'vs', config['vs_python'], 'vectorize_strokes_batch.py',
                    ['--manifest', str(export / 'manifest.jsonl'), '--repo', config['vs_repo'],
                     '--model', config['vs_model'], '--samples', str(config['vs_samples']),
                     '--seed', str(config['vs_seed']), '--start-cursor-source', config['start_cursor_source'],
                     '--vs-device', 'cpu', '--mmah-manifest', str(references)],
                    ['manifest.jsonl', 'errors.jsonl', 'model_config.json'],
                    partial_count=exported_count * config['vs_samples'], resumable=True)
        raw_count = sum(1 for _ in iter_records(raw / 'manifest.jsonl'))
        qc = stage(directory, 'qc', python, 'render_and_filter.py',
                   ['--manifest', str(raw / 'manifest.jsonl'), '--iou', str(threshold),
                    '--fidelity', str(threshold)],
                   ['manifest.jsonl', 'best.jsonl', 'errors.jsonl', 'summary.json'], partial_count=raw_count)
        chars = stage(directory, 'assemble', python, 'assemble_vs_exploratory.py',
                      ['--manifest', str(qc / 'best.jsonl'), '--iou', str(threshold)],
                      ['manifest.jsonl', 'rejected.jsonl', 'summary.json'])
        stroke_manifest, character_manifest = qc / 'best.jsonl', chars / 'manifest.jsonl'
    else:
        empty = directory / 'empty.jsonl'
        write_records(empty, [])
        stroke_manifest = character_manifest = empty
    dataset = Path(tempfile.mkdtemp(prefix='dataset-', dir=str(directory)))
    finalize_manifest(rows, stroke_manifest, character_manifest, cleanup_report,
                      dataset, threshold, config['trajectory_style'], copy_assets)
    write_json(marker, dict(output=str(dataset), hashes={
        name: digest(dataset / name) for name in ('manifest.jsonl', 'summary.json')}))
    print('[batch done] ' + json.dumps(read_json(dataset / 'summary.json'), ensure_ascii=False), flush=True)
    return dataset / 'manifest.jsonl'


def merge_datasets(manifests, source_manifest, output, threshold, trajectory_style, copy_assets):
    output.mkdir(parents=True, exist_ok=True)
    # Disk-backed lookup bounds memory even when the final manifest is gigabytes.
    dbfile = output / 'merge.sqlite'
    with sqlite3.connect(str(dbfile)) as db:
        db.execute('DROP TABLE IF EXISTS samples')
        db.execute('CREATE TABLE samples (id TEXT PRIMARY KEY, body TEXT NOT NULL)')
        for manifest in manifests:
            for row in iter_records(manifest):
                db.execute('INSERT INTO samples VALUES (?, ?)',
                           (row['sample_id'], json.dumps(row, ensure_ascii=False, allow_nan=False)))
            db.commit()
        count = Counter()
        def merged_rows():
            for source in iter_records(source_manifest):
                found = db.execute('SELECT body FROM samples WHERE id=?', (source['sample_id'],)).fetchone()
                row = json.loads(found[0]) if found else pending_record(source, threshold, trajectory_style)
                if copy_assets and row['build_status'] == 'trajectory_pending':
                    destination = output / 'pending_samples' / source['sample_id']
                    row['image_path'] = copy_file(source['image_path'], destination / 'image_256.png')
                    if source.get('source_image_path'):
                        original = Path(source['source_image_path'])
                        row['original_image_path'] = copy_file(original, destination / ('original_calligraphy' + original.suffix))
                count[row['build_status']] += 1
                count['quality_passed'] += bool(row['quality']['passed'])
                yield row
        write_records(output / 'manifest.jsonl', merged_rows())
        write_json(output / 'summary.json', dict(samples=count['trajectory_ready'] + count['trajectory_pending'],
                   ready=count['trajectory_ready'], pending=count['trajectory_pending'],
                   quality_passed=count['quality_passed'], threshold=threshold,
                   note='Batch dataset assets remain under the continuation directory. No human approval claimed.'))
    dbfile.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='Existing run with completed mask_cleanup')
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--output', help='New or resumable continuation directory; default RUN/continuation_v1')
    p.add_argument('--max-batches', type=int, help='Optional limit for a small real validation run')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    require(1 <= a.batch_size <= 500, 'batch-size must be 1..500')
    require(a.max_batches is None or a.max_batches > 0, 'max-batches must be positive')
    run = Path(a.run).resolve()
    output = Path(a.output).resolve() if a.output else run / 'continuation_v1'
    require(output != run and output not in run.parents and not any(
        output == run / name or run / name in output.parents
        for name in ('selected', 'extracted', 'mask_cleanup')),
        'Use a separate continuation directory')
    cleanup = run / 'mask_cleanup/manifest.jsonl'
    references = run / 'selected/references.jsonl'
    sources = run / 'selected/all_sources.jsonl'
    config = read_json(run / 'run_config.json')
    require(config['route'] == 'vs' and config['vs_device'] == 'cpu',
            'This continuation preserves the existing VS CPU route only')
    for path in (cleanup, references, sources):
        require(path.is_file(), 'Missing prerequisite: ' + str(path))
    require((run / 'mask_cleanup/errors.jsonl').is_file() and
            (run / 'mask_cleanup/errors.jsonl').stat().st_size == 0,
            'Cleanup has errors or incomplete accounting; inspect before continuing')
    summary = cleanup_summary(run / 'mask_cleanup/report.json')
    require(summary['errors'] == 0 and summary['processed'] == summary['input_samples'],
            'Cleanup did not finish all extracted samples')
    threshold = config['quality_threshold']
    if a.dry_run:
        first = list(islice(iter_records(cleanup), a.batch_size))
        require(first, 'Empty cleanup manifest')
        for row in first:
            require(Path(row['extracted_path']).is_file(), 'Missing cleaned NPZ')
            require(read_json(row['cleanup_report_path'])['sample_id'] == row['sample_id'], 'Report mismatch')
        print('Validated first batch: %d samples. No inference executed.' % len(first))
        print('Plan: stream cleanup manifest -> batches of %d -> export -> VS CPU -> QC -> assemble -> merge' % a.batch_size)
        print('Completed cleanup: %d samples. Resume output: %s' % (summary['processed'], output))
        return
    modeldir = Path(config['vs_repo']) / 'outputs/snapshot' / config['vs_model']
    assets = list(modeldir.glob('*.index')) + list(modeldir.glob('*.data-*')) + [modeldir / 'model_config.json']
    require(len(assets) >= 3, 'VS checkpoint missing')
    assets += [Path(__file__).resolve(), ROOT / 'CHTR/data/build_chtr_dataset.py']
    assets += [PIPELINE / name for name in ('common.py', 'export_clean_masks_for_vs.py',
               'vectorize_strokes_batch.py', 'render_and_filter.py', 'decode_vs_npz.py', 'assemble_vs_exploratory.py')]
    assets += list(Path(config['vs_repo']).glob('*.py')) + [Path(config['vs_repo']) / 'tools/visualize_drawing.py']
    signature = dict(batch_size=a.batch_size, config=config,
                     hashes={str(path): digest(path) for path in [cleanup, references, sources] + assets})
    with exclusive_run(output):
        state = output / 'config.json'
        if state.exists():
            require(read_json(state) == signature, 'Inputs/config/code changed; use a new --output directory')
        else:
            write_json(state, signature)
        batches = output / 'batches'; batches.mkdir(exist_ok=True)
        manifests = []
        processed = 0
        for index, rows in enumerate(chunks(iter_records(cleanup), a.batch_size)):
            if a.max_batches is not None and index >= a.max_batches:
                print('[limited run] completed %d batches; final merge deferred' % index, flush=True)
                return
            directory = batches / ('batch_%06d' % index)
            manifest = run_batch(directory, rows, config, threshold, references, config['copy_assets'])
            manifests.append(manifest); processed += len(rows)
            print('[progress] batches=%d cleaned_samples=%d' % (index + 1, processed), flush=True)
        require(processed == sum(1 for _ in iter_records(run / 'extracted/manifest.jsonl')),
                'Cleanup does not cover all successful extractions')
        merge_datasets(manifests, sources, output / 'dataset', threshold, config['trajectory_style'], config['copy_assets'])
        print('[done] ' + str(output / 'dataset/summary.json'), flush=True)


if __name__ == '__main__':
    main()
