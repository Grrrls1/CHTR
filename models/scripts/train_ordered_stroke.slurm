#!/bin/bash
#SBATCH --job-name=chtr-ordered-stroke
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/home/lkh/workspace/CHTR
#SBATCH --output=/home/lkh/workspace/logs/%x-%j.out
#SBATCH --error=/home/lkh/workspace/logs/%x-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
: "${CUDA_VISIBLE_DEVICES:?Slurm must allocate a GPU}"
: "${SLURM_JOB_ID:?Submit with gpu-sbatch}"
TRAIN_PY=/home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python
TRAIN_ROOT="models/runs/ordered_stroke_full/job_${SLURM_JOB_ID}"
[[ -x "$TRAIN_PY" ]]
[[ -f data/chtr_mmah_lmdb/complete.json ]]
[[ ! -e "$TRAIN_ROOT" ]]
exec 9>/home/lkh/workspace/CHTR/data/.chtr_mmah_lmdb.lock
flock -s 9
mkdir -p "$TRAIN_ROOT"
cp models/configs/ordered_stroke.json "$TRAIN_ROOT/input_config.json"
cp models/scripts/train_ordered_stroke.slurm "$TRAIN_ROOT/job.slurm"
srun "$TRAIN_PY" models/scripts/check_environment.py \
  --require-cuda --require-torchvision --device cuda:0 --output-json "$TRAIN_ROOT/environment.json"
srun "$TRAIN_PY" models/scripts/train_ordered_stroke.py \
  --config "$TRAIN_ROOT/input_config.json" --output-dir "$TRAIN_ROOT/train" --device cuda:0
