#!/bin/bash
#SBATCH --job-name=chtr-stroke-full
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
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
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
: "${CUDA_VISIBLE_DEVICES:?Slurm must allocate a GPU}"
: "${SLURM_JOB_ID:?Submit with gpu-sbatch}"

TRAIN_PY=/home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python
TRAIN_ROOT="models/runs/single_stroke_full/job_${SLURM_JOB_ID}"
[[ -x "$TRAIN_PY" ]]
[[ -f data/chtr_mmah_lmdb/complete.json ]]
[[ ! -e "$TRAIN_ROOT" ]] || { echo "Output already exists: $TRAIN_ROOT" >&2; exit 2; }
exec 9>/home/lkh/workspace/CHTR/data/.chtr_mmah_lmdb.lock
flock -s 9
mkdir -p "$TRAIN_ROOT"
cp models/configs/single_stroke.json "$TRAIN_ROOT/input_config.json"
cp models/scripts/train_single_stroke_full.slurm "$TRAIN_ROOT/job.slurm"
printf 'Job=%s\nPython=%s\nOutput=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$SLURM_JOB_ID" "$TRAIN_PY" "$TRAIN_ROOT" "$CUDA_VISIBLE_DEVICES"

# Environment diagnostics only; launch full training directly without a smoke run.
srun "$TRAIN_PY" models/scripts/check_environment.py \
  --require-cuda --require-torchvision --device cuda:0 \
  --output-json "$TRAIN_ROOT/environment.json"
srun "$TRAIN_PY" models/scripts/train_single_stroke.py \
  --config "$TRAIN_ROOT/input_config.json" \
  --output-dir "$TRAIN_ROOT/train" --device cuda:0
