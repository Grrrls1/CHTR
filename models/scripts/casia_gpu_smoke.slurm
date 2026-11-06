#!/bin/bash
#SBATCH --job-name=chtr-casia-smoke
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/lkh/workspace/CHTR
#SBATCH --output=/home/lkh/workspace/logs/%x-%j.out
#SBATCH --error=/home/lkh/workspace/logs/%x-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with gpu-sbatch --user_name lkh}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must allocate a GPU}"
[[ $(id -un) == lkh ]]
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=/home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python
ROOT="models/runs/casia_olhwdb11/smoke/job_${SLURM_JOB_ID}"
mkdir -p "$ROOT"
cp models/scripts/casia_gpu_smoke.slurm "$ROOT/job.slurm"
srun "$PY" models/scripts/check_environment.py --require-cuda --require-torchvision --device cuda:0 --output-json "$ROOT/environment.json"
srun --unbuffered "$PY" -u models/scripts/casia_gpu_smoke.py --output "$ROOT/check"
