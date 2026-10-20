#!/usr/bin/env bash
# Single-GPU training; CUDA_VISIBLE_DEVICES maps the selected GPU to cuda:0.
set -euo pipefail
model_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${CHTR_PYTHON:-$model_root/.venv/bin/python}"
if [[ -z "${CHTR_PYTHON:-}" && ! -x "$task_python" ]]; then
  task_python="$model_root/../third_party/StrokeExtraction/.venv/bin/python"
fi
gpu="${CUDA_VISIBLE_DEVICES:-0}"
usage() {
  cat <<'HELP'
Usage: bash train_gpu.sh [--python PATH] [--gpu ID_OR_UUID] [train.py options]
Example: bash train_gpu.sh --gpu 2 --max-steps 4
Default: CHTR_PYTHON, models/.venv if present, then third_party/StrokeExtraction/.venv.
Preserves existing CUDA_VISIBLE_DEVICES or uses 0.
This is single-GPU training, not distributed training.
HELP
}
while (($#)); do
  case "$1" in
    --python|--gpu)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in --python) task_python="$2" ;; --gpu) gpu="$2" ;; esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
"$task_python" "$model_root/scripts/check_environment.py" --require-cuda --device cuda:0
exec "$task_python" "$model_root/scripts/train.py" \
  --config "$model_root/configs/gpu.json" "$@" --device cuda:0
