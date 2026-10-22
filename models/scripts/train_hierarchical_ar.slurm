#!/bin/bash
#SBATCH --job-name=chtr-ar3-r18-b24
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with gpu-sbatch --user_name YOUR_NAME}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must allocate a GPU}"
project_root="$SLURM_SUBMIT_DIR"
[[ -f "$project_root/models/scripts/train_hierarchical_ar.py" ]] || project_root="$project_root/CHTR"
project_root="$(cd -- "$project_root" && pwd -P)"
train_python="${CHTR_PYTHON:-$project_root/third_party/StrokeExtraction/.venv/bin/python}"
train_output="$project_root/models/runs/hierarchical_ar_three_task/job_$SLURM_JOB_ID"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec 9>"$project_root/data/.chtr_mmah_lmdb.lock"
flock -s 9
# Resume after a scheduler requeue uses this new mode's own checkpoint only.
resume_args=()
if [[ -f "$train_output/last.pt" ]]; then resume_args=(--resume "$train_output/last.pt"); fi
train_step_pid=''
trap '[[ -z "$train_step_pid" ]] || kill -USR1 "$train_step_pid" 2>/dev/null || true' USR1
srun "$train_python" "$project_root/models/scripts/train_hierarchical_ar.py" \
  --config "$project_root/models/configs/hierarchical_ar.json" --output-dir "$train_output" "${resume_args[@]}" "$@" &
train_step_pid=$!
set +e
wait "$train_step_pid"
train_exit=$?
if kill -0 "$train_step_pid" 2>/dev/null; then wait "$train_step_pid";train_exit=$?;fi
exit "$train_exit"
