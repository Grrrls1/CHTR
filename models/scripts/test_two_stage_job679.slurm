#!/bin/bash
#SBATCH --job-name=chtr-two-stage-test679
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/home/lkh/workspace/CHTR
#SBATCH --output=/home/lkh/workspace/logs/%x-%j.out
#SBATCH --error=/home/lkh/workspace/logs/%x-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
: "${CUDA_VISIBLE_DEVICES:?Slurm must allocate a GPU}"
: "${SLURM_JOB_ID:?Submit with gpu-sbatch}"
EVAL_PY=/home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python
STROKE_CHECKPOINT=models/runs/ordered_stroke_full/job_679/train/ordered_stroke.pt
POINT_CHECKPOINT=models/runs/single_stroke_full/job_605/train/single_stroke.pt
OUTPUT="models/runs/ordered_stroke_full/job_679/test_job_${SLURM_JOB_ID}"
[[ -x "$EVAL_PY" ]]
[[ -f "$STROKE_CHECKPOINT" ]]
[[ -f "$POINT_CHECKPOINT" ]]
[[ ! -e "$OUTPUT" ]]
printf 'Job=%s\nStrokeCheckpoint=%s\nPointCheckpoint=%s\nOutput=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$SLURM_JOB_ID" "$STROKE_CHECKPOINT" "$POINT_CHECKPOINT" "$OUTPUT" "$CUDA_VISIBLE_DEVICES"
exec 9>/home/lkh/workspace/CHTR/data/.chtr_mmah_lmdb.lock
flock -s 9
srun --unbuffered "$EVAL_PY" -u models/scripts/evaluate_ordered_stroke.py \
  --checkpoint "$STROKE_CHECKPOINT" --point-checkpoint "$POINT_CHECKPOINT" \
  --output "$OUTPUT" --split test --limit 0 \
  --batch-size 2 --point-batch-size 16 --device cuda:0
