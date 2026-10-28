#!/bin/bash
#SBATCH --job-name=chtr-stroke-test605
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
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

EVAL_PY=/home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python
CHECKPOINT=models/runs/single_stroke_full/job_605/train/best.pt
OUTPUT="models/runs/single_stroke_full/job_605/test_job_${SLURM_JOB_ID}"
[[ -x "$EVAL_PY" ]]
[[ -f "$CHECKPOINT" ]]
[[ ! -e "$OUTPUT" ]]
printf 'Job=%s\nCheckpoint=%s\nOutput=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$SLURM_JOB_ID" "$CHECKPOINT" "$OUTPUT" "$CUDA_VISIBLE_DEVICES"
echo 'Acquiring shared dataset lock...'
exec 9>/home/lkh/workspace/CHTR/data/.chtr_mmah_lmdb.lock
flock -s 9
echo 'Dataset lock acquired; launching test.'

srun --unbuffered "$EVAL_PY" -u models/scripts/evaluate_single_stroke.py \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --split test --limit 0 \
  --batch-size 32 --visualize-characters 30 \
  --threshold 0.5 --device cuda:0 --threads 4
