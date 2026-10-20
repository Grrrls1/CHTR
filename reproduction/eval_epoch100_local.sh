#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CHTR_PYTHON:-$ROOT/../.venv-chtr-eval/bin/python}"
DEVICE="${CHTR_DEVICE:-cuda:0}"
DATASET="${CHTR_DATASET:-$ROOT/data/chtr_mmah_lmdb}"
OUTPUT="${CHTR_OUTPUT:-$ROOT/models/runs/full_separated_10gb/job_237/eval_epoch_100_$(date +%Y%m%d_%H%M%S)}"

if [[ -e "$OUTPUT" ]]; then
  echo "输出目录已存在，请设置新的 CHTR_OUTPUT：$OUTPUT" >&2
  exit 1
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.cache/matplotlib}"
"$PYTHON" - "$DEVICE" <<'PY'
import sys
import cv2, lmdb, matplotlib, numpy, torch
if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    raise SystemExit('评估解码需要 PyTorch >= 2.0，请通过 CHTR_PYTHON 指定兼容环境。')
device = sys.argv[1]
if device.startswith('cuda'):
    if not torch.cuda.is_available():
        raise SystemExit('CUDA 不可用。请检查 NVIDIA 驱动；CPU 运行可设置 CHTR_DEVICE=cpu。')
    torch.empty(1, device=device)
print(f'Python: {sys.executable}; torch: {torch.__version__}; device: {device}', flush=True)
PY
cd "$ROOT"
exec "$PYTHON" -u models/scripts/evaluate_memorization.py \
  --checkpoint "$ROOT/models/runs/full_separated_10gb/job_237/last.pt" \
  --dataset "$DATASET" --device "$DEVICE" \
  --limit "${CHTR_LIMIT:-32}" --max-strokes 64 --max-points 1024 \
  --output "$OUTPUT"
