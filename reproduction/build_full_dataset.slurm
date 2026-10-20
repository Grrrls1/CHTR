#!/bin/bash
#SBATCH --job-name=chtr-full-data
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Submit from the workspace/release root with gpu-sbatch --user_name YOUR_NAME.
set -euo pipefail
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
build_root="${CHTR_WORKSPACE:-${SLURM_SUBMIT_DIR:-$source_root}}"
if [[ -f "$build_root/data/build_chtr_dataset.py" ]]; then
  build_root="$(cd -- "$build_root/.." && pwd -P)"
fi
vendor_root="$build_root/CHTR/third_party"
build_python="${CHTR_PYTHON:-$vendor_root/StrokeExtraction/.venv/bin/python}"
runner=()
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  : "${CUDA_VISIBLE_DEVICES:?Slurm must allocate a GPU}"
  runner=(srun)
else
  readonly_only=false
  for argument in "$@"; do
    case "$argument" in --dry-run|--help|-h) readonly_only=true ;; esac
  done
  [[ "$readonly_only" == true ]] || { echo 'Submit computation through Slurm; direct execution supports --help/--dry-run only.' >&2; exit 2; }
fi
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${runner[@]}" "$build_python" - "$build_root" "$vendor_root" "$@" <<'PY'
"""Extract/clean -> bounded skeleton batches -> all-clean LMDB, with one entry."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT, VENDOR = map(Path, sys.argv[1:3])
DATA = ROOT/'CHTR/data'
PIPE = ROOT/'CHTR/reproduction/pipeline'
RUNS = ROOT/'CHTR/reproduction/pipeline_runs'
LEGACY_MAP = RUNS/'mmah_single_path_gpu2_20261005_five/dataset_simplified_dataloader_v2/stroke_type_map.json'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', default=str(RUNS/'calligraphy_full_gpu0_20261013_134511_2134141'))
p.add_argument('--dataset', default=str(DATA/'chtr_mmah_lmdb'))
p.add_argument('--pack-output', help='New repack cache directory when historical code fingerprints differ')
p.add_argument('--source-manifest', default=str(ROOT/'datasets/processed/chinese-calligraphy-221030/manifest.jsonl'))
p.add_argument('--mmah-bank', default=str(RUNS/'mmah_reference_bank_20260918/manifest.jsonl'))
p.add_argument('--stroke-type-map', default=str(ROOT/'configs/stroke_type_map.json' if (ROOT/'configs/stroke_type_map.json').is_file() else LEGACY_MAP))
p.add_argument('--from-stage', choices=('prepare', 'skeleton', 'pack'), default='prepare')
p.add_argument('--quality-threshold', type=float, default=.87)
p.add_argument('--batch-size', type=int, default=100)
p.add_argument('--gpu-batch-size', type=int, default=64)
p.add_argument('--workers', type=int, default=3)
p.add_argument('--max-points', type=int, default=1024)
p.add_argument('--dry-run', action='store_true', help='Print the complete plan without writing files or running computation')
a = p.parse_args(sys.argv[3:])
if not (0 <= a.quality_threshold <= 1 and 1 <= a.batch_size <= 500 and 1 <= a.gpu_batch_size <= 256 and 1 <= a.workers <= 16 and a.max_points == 1024):
    p.error('Invalid limits; this full-data packing route requires max-points=1024')
run = Path(a.run).resolve()
dataset = Path(a.dataset).resolve()
config_path = run/'run_config.json'
selected, extracted, cleanup = (run/n for n in ('selected', 'extracted', 'mask_cleanup'))
skeleton = run/'skeleton_continuation_v1'
read = lambda path: json.loads(path.read_text())
def require(condition, message):
    if not condition:
        raise SystemExit(message)
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
def cleanup_header():
    path = cleanup/'report.json'
    if not path.is_file():
        return None
    header = ''
    with path.open() as stream:
        for line in stream:
            if line.strip() == '"samples": [':
                return json.loads(header+'"samples": []}')
            header += line
            require(len(header) < 65536, 'Invalid cleanup report header')
    raise SystemExit('Incomplete cleanup report')
def command(script, *arguments, partial=False):
    cmd = [sys.executable, str(script), *map(str, arguments)]
    print(shlex.join(cmd), flush=True)
    if a.dry_run:
        return
    code = subprocess.call(cmd)
    if partial and code == 1:
        require((extracted/'manifest.jsonl').is_file() and (extracted/'errors.jsonl').is_file(), 'Extraction failed before producing partial results')
    else:
        require(code == 0, 'Stage failed with exit code '+str(code))

require((DATA/'build_chtr_dataset.py').is_file(), 'Submit from the workspace/release root, or set CHTR_WORKSPACE')
if not a.dry_run:
    run.mkdir(parents=True, exist_ok=True)
    lock = (run/'.full_build.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(Path(a.stroke_type_map).is_file(), 'Missing stroke type map')
config = read(config_path) if config_path.is_file() else None
if config is not None:
    require(config['quality_threshold'] == a.quality_threshold, 'Existing cleanup threshold differs; use a new run')
header = cleanup_header()
clean_complete = header is not None and header['processed'] == header['input_samples'] and header['errors'] == 0 and (cleanup/'manifest.jsonl').is_file() and (cleanup/'errors.jsonl').is_file() and (cleanup/'errors.jsonl').stat().st_size == 0
if a.from_stage == 'prepare' and not clean_complete:
    print('[1/3] Select all matching images, extract strokes, clean masks', flush=True)
    if not a.dry_run:
        sys.path.insert(0, str(DATA))
        from build_chtr_dataset import select_inputs
        if config is None:
            require(not any(run.iterdir()) or set(x.name for x in run.iterdir()) == {'.full_build.lock'}, 'Use a fresh run, or one with run_config.json')
            selected.mkdir()
            select_inputs(a.source_manifest, a.mmah_bank, selected)
            config = dict(source_manifest=str(Path(a.source_manifest).resolve()), mmah_bank=str(Path(a.mmah_bank).resolve()),
                output=str(run), route='skeleton', quality_threshold=a.quality_threshold, trajectory_style='MMAH宋体',
                pipeline_python=sys.executable, stroke_python=sys.executable, copy_assets=True,
                stroke_repo=str(VENDOR/'StrokeExtraction'), allow_extraction_errors=True)
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
        require(not cleanup.exists(), 'Incomplete cleanup exists; inspect it and use a new run rather than overwriting')
    # Preserve historical config/signatures, but invoke the relocated dependency.
    repo = VENDOR/'StrokeExtraction'
    weights = []
    for flag, folder, filename in [('--sdnet-model','SDNet','sdnet_model.pth'),('--segnet-model','SegNet','model.pth'),('--extractnet-model','ExtractNet','model_extract.pth')]:
        key = flag[2:].replace('-', '_')
        weight = config.get(key, str(repo/'out'/(folder+'_MMAH_CCSEDB_v4')/'model'/filename)) if config else str(repo/'out'/(folder+'_MMAH_CCSEDB_v4')/'model'/filename)
        for name in ('StrokeExtraction', 'virtual_sketching'):
            old = str(ROOT/name)
            if weight.startswith(old+'/'):
                weight = str(VENDOR/name)+weight[len(old):]
        weights += [flag, weight]
    command(PIPE/'extract_strokes_batch.py', '--manifest', selected/'samples.jsonl', '--references', selected/'references.jsonl',
        '--output', extracted, '--repo', repo, '--gpu', os.environ.get('CUDA_VISIBLE_DEVICES','0'), '--device', 'cuda',
        *weights, *(['--resume'] if extracted.exists() else []), partial=True)
    command(PIPE/'clean_stroke_masks.py', '--manifest', extracted/'manifest.jsonl', '--output', cleanup, '--whole-iou', a.quality_threshold)
elif clean_complete:
    print('[1/3] Reuse completed extraction/cleanup', flush=True)
elif not a.dry_run:
    raise SystemExit('Cleanup is not complete; start with --from-stage prepare')

parameters = dict(batch_size=a.batch_size, gpu_batch_size=a.gpu_batch_size, cpu_workers=a.workers, device='cuda', max_points=a.max_points)
expected_batches = math.ceil(header['processed']/a.batch_size) if clean_complete else 0
completed = (skeleton/'dataset/training_summary.json').is_file() and (skeleton/'dataset/manifest.jsonl').is_file() and expected_batches > 0 and len(list((skeleton/'batches').glob('batch_*'))) == expected_batches and all((skeleton/'batches'/('batch_%06d'%i)/'dataset.done.json').is_file() for i in range(expected_batches))
if a.from_stage != 'pack' and completed:
    previous = read(skeleton/'config.json')
    require(previous['parameters'] == parameters and previous['config'] == config, 'Completed skeleton settings differ; use a new run')
    if not a.dry_run:
        for path in (cleanup/'manifest.jsonl', selected/'references.jsonl', selected/'all_sources.jsonl'):
            require(previous['hashes'][str(path)] == digest(path), 'Skeleton source changed: '+str(path))
    print('[2/3] Reuse completed skeleton batches and merge; do not rewrite historical fingerprints', flush=True)
elif a.from_stage != 'pack':
    print('[2/3] Bounded GPU thinning and CPU tracing/QC; defer final LMDB to stage 3', flush=True)
    command(DATA/'continue_skeleton_dataset.py', '--run', run, '--device', 'cuda', '--batch-size', a.batch_size,
        '--gpu-batch-size', a.gpu_batch_size, '--cpu-workers', a.workers, '--max-points', a.max_points,
        '--stroke-type-map', a.stroke_type_map)
elif not a.dry_run:
    require(completed, 'Packing requires all skeleton batches and the final merge to be complete')
print('[3/3] Repack every cleanup-qualified image; retain pending records; publish LMDB with backup', flush=True)
packed = Path(a.pack_output).resolve() if a.pack_output else run/'skeleton_all_clean_v1'
if not a.dry_run and (dataset/'complete.json').is_file() and (packed/'manifest.jsonl').is_file():
    published = read(dataset/'complete.json').get('signature', {}).get('manifest_sha256')
    if published == digest(packed/'manifest.jsonl'):
        print('[complete] Reuse already published LMDB; historical fingerprints left unchanged: '+str(dataset), flush=True)
        raise SystemExit(0)
command(DATA/'repack_all_clean.py', '--run', run, '--workers', a.workers, '--stroke-type-map', a.stroke_type_map,
    '--dataset', dataset, '--publish', *(['--output', str(packed)] if a.pack_output else []))
print('[dry-run complete]' if a.dry_run else '[complete] '+str(dataset), flush=True)
PY
