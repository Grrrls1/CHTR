#!/bin/bash
#SBATCH --job-name=chtr-stroke-smoke
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/lkh/workspace/CHTR
#SBATCH --output=/home/lkh/workspace/logs/%x-%j.out
#SBATCH --error=/home/lkh/workspace/logs/%x-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Only load the trusted checkpoints created by this job.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
: "${CUDA_VISIBLE_DEVICES:?Slurm must allocate a GPU}"
: "${SLURM_JOB_ID:?Submit with gpu-sbatch}"

SMOKE_PY=/home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python
SMOKE_CONFIG=models/configs/single_stroke_smoke.json
SMOKE_ROOT="models/runs/single_stroke_gpu_smoke/job_${SLURM_JOB_ID}"
SMOKE_TRAIN="$SMOKE_ROOT/train"
[[ ! -e "$SMOKE_ROOT" ]] || { echo "Output already exists: $SMOKE_ROOT" >&2; exit 2; }
[[ -x "$SMOKE_PY" ]]
[[ -f data/chtr_mmah_lmdb/complete.json ]]
exec 9>/home/lkh/workspace/CHTR/data/.chtr_mmah_lmdb.lock
flock -s 9
mkdir -p "$SMOKE_ROOT"
cp "$SMOKE_CONFIG" "$SMOKE_ROOT/input_config.json"
SMOKE_CONFIG="$SMOKE_ROOT/input_config.json"
# Keep this archived diagnostic short; explicit partial evaluation follows below.
"$SMOKE_PY" - "$SMOKE_CONFIG" <<'CONFIG'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);c=json.loads(p.read_text());c['eval_every_epochs']=10
p.write_text(json.dumps(c,indent=2)+'\n')
CONFIG
cp models/scripts/single_stroke_gpu_smoke.slurm "$SMOKE_ROOT/job.slurm"
printf 'Job=%s\nPython=%s\nOutput=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$SLURM_JOB_ID" "$SMOKE_PY" "$SMOKE_ROOT" "$CUDA_VISIBLE_DEVICES"

srun "$SMOKE_PY" models/scripts/check_environment.py \
  --require-cuda --require-torchvision --device cuda:0 \
  --output-json "$SMOKE_ROOT/environment.json"

# Eight length-spread training strokes from 32 training characters; two val strokes.
srun "$SMOKE_PY" models/scripts/train_single_stroke.py \
  --config "$SMOKE_CONFIG" --output-dir "$SMOKE_TRAIN" --device cuda:0 --max-steps 4
cp "$SMOKE_TRAIN/training_status.json" "$SMOKE_ROOT/status_step4.json"
srun "$SMOKE_PY" models/scripts/train_single_stroke.py \
  --config "$SMOKE_CONFIG" --output-dir "$SMOKE_TRAIN" --device cuda:0 \
  --resume "$SMOKE_TRAIN/last.pt" --max-steps 8

# Label-conditioned validation only; test is not used for model selection.
srun "$SMOKE_PY" models/scripts/evaluate_single_stroke.py \
  --checkpoint "$SMOKE_TRAIN/last.pt" --output "$SMOKE_ROOT/eval_val" \
  --split val --limit 3 --decode-max-points 64 --device cuda:0

srun "$SMOKE_PY" - "$SMOKE_ROOT" <<'PY'
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
read = lambda name: json.loads((root / name).read_text())
env = read('environment.json')
status4 = read('status_step4.json')
status = read('train/training_status.json')
usage = read('train/data_usage.json')
evaluation = read('eval_val/summary.json')
metrics = [json.loads(line) for line in (root / 'train/metrics.jsonl').read_text().splitlines()]
training = [row for row in metrics if row['kind'] == 'train']
assert env['ok'] and env['cuda']['kernel_and_backward_check'] == 'passed'
assert status4['step'] == 4 and status['step'] == 8 and status['device'] == 'cuda:0'
assert [row['step'] for row in training] == list(range(1, 9))
assert all(math.isfinite(value) for row in metrics for value in row.values()
           if isinstance(value, (int, float)))
assert (root / 'train/last.pt').is_file()
assert usage['evaluation_split'] == 'val' and not usage['test_used_for_model_selection']
assert evaluation['split'] == 'val' and evaluation['strokes'] == 3
assert all(json.loads(line)['first_xyz_exact'] for line in (root / 'eval_val/strokes.jsonl').read_text().splitlines())
assert (root / 'eval_val/index.html').is_file()
assert len(list((root / 'eval_val').rglob('*.npz'))) == 3
report = dict(passed=True, job_id=root.name.removeprefix('job_'),
              device=env['cuda'], training_status=status,
              train_strokes=usage['train_strokes'], validation_exports=3,
              first_train_loss=training[0]['loss'], last_train_loss=training[-1]['loss'],
              scope='GPU kernels/backward, 8 training steps, validation, resume and conditional export; not convergence or full-size capacity.')
(root / 'smoke_result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
PY
