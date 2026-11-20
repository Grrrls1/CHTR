#!/bin/bash
#SBATCH --job-name=hiwtr-sampled-n100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/home/lkh/workspace/CHTR
#SBATCH --output=/home/lkh/workspace/logs/%x-%j.out
#SBATCH --error=/home/lkh/workspace/logs/%x-%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
: "${SLURM_JOB_ID:?Submit with gpu-sbatch --user_name lkh}"
: "${CUDA_VISIBLE_DEVICES:?GPU allocation required}"
MODE=${1:?smoke or run}
OUTPUT=${2:?frozen output directory}
exec 9>"$OUTPUT/.evaluation.lock"
flock -n 9
exec 8>/home/lkh/workspace/CHTR/data/.chtr_mmah_lmdb.lock
flock -s 8
srun --unbuffered third_party/StrokeExtraction/.venv/bin/python -u models/scripts/evaluate_cross_domain_sampled.py "$MODE" --output "$OUTPUT"
