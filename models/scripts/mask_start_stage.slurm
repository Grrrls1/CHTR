#!/bin/bash
#SBATCH --job-name=hiwtr-mask-start
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/lkh/workspace/CHTR
#SBATCH --output=/home/lkh/workspace/logs/%x-%A_%a.out
#SBATCH --error=/home/lkh/workspace/logs/%x-%A_%a.err
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
: "${SLURM_JOB_ID:?Submit with gpu-sbatch --user_name lkh}"
: "${CUDA_VISIBLE_DEVICES:?GPU allocation required}"
MODE=${1:?smoke/base/noise}
OUTPUT=${2:?new frozen output directory}
SPLIT=${3:-}
exec 9>/home/lkh/workspace/CHTR/data/.chtr_mmah_lmdb.lock
flock -s 9
ARGS=("$MODE" --output "$OUTPUT")
if [[ "$MODE" == noise ]]; then
  : "${SLURM_ARRAY_TASK_ID:?Noise uses 12 fixed point/mask/strength partitions}"
  ARGS+=(--split "$SPLIT" --partition "$SLURM_ARRAY_TASK_ID")
fi
srun --unbuffered third_party/StrokeExtraction/.venv/bin/python -u models/scripts/evaluate_mask_start_stage.py "${ARGS[@]}"
