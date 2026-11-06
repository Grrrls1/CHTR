#!/bin/bash
#SBATCH --job-name=chtr-casia
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/home/lkh/workspace/CHTR
#SBATCH --output=/home/lkh/workspace/logs/%x-%j.out
#SBATCH --error=/home/lkh/workspace/logs/%x-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through gpu-sbatch --user_name lkh}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must allocate a GPU}"
[[ $(id -un) == lkh ]]
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=/home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python
STAGE="${1:?single_stroke or ordered_stroke}"; shift
[[ "$STAGE" == single_stroke || "$STAGE" == ordered_stroke ]]
CONFIG="${1:-models/configs/casia_${STAGE}.json}"
if (( $# )); then shift; fi
ROOT="models/runs/casia_olhwdb11/${STAGE}/job_${SLURM_JOB_ID}"
[[ ! -e "$ROOT" ]]
mkdir -p "$ROOT"
cp "$CONFIG" "$ROOT/input_config.json"
cp models/scripts/train_casia.slurm "$ROOT/job.slurm"
# The ordered stage requires the best CASIA point checkpoint path in its config.
srun "$PY" models/scripts/check_environment.py --require-cuda --require-torchvision --device cuda:0 --output-json "$ROOT/environment.json"
srun --unbuffered "$PY" -u "models/scripts/train_${STAGE}.py" --config "$ROOT/input_config.json" --output-dir "$ROOT/train" --device cuda:0 "$@"
