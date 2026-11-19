#!/bin/bash
#SBATCH --job-name=chtr-calli-point-ft
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

TRAIN_PY=/home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python
TRAIN_CONFIG="${1:-models/configs/calligraphy_single_stroke_finetune.json}"
TRAIN_ROOT="models/runs/casia_to_calligraphy/single_stroke/job_${SLURM_JOB_ID}"
[[ -x "$TRAIN_PY" && -f "$TRAIN_CONFIG" && -f data/chtr_mmah_lmdb/complete.json ]]
[[ ! -e "$TRAIN_ROOT" ]] || { echo "Output already exists: $TRAIN_ROOT" >&2; exit 2; }
exec 9>/home/lkh/workspace/CHTR/data/.chtr_mmah_lmdb.lock
flock -s 9
mkdir -p "$TRAIN_ROOT"
cp "$TRAIN_CONFIG" "$TRAIN_ROOT/input_config.json"
cp models/configs/casia_single_stroke.json "$TRAIN_ROOT/casia_reference_config.json"
cp models/scripts/finetune_calligraphy_point.slurm "$TRAIN_ROOT/job.slurm"

srun "$TRAIN_PY" models/scripts/check_environment.py \
  --require-cuda --require-torchvision --device cuda:0 \
  --output-json "$TRAIN_ROOT/environment.json"
srun --unbuffered "$TRAIN_PY" -u models/scripts/train_single_stroke.py \
  --config "$TRAIN_ROOT/input_config.json" --output-dir "$TRAIN_ROOT/train" --device cuda:0
