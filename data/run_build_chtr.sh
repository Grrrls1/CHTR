#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE_PY="${CHTR_PIPELINE_PY:-$ROOT_DIR/CHTR/third_party/StrokeExtraction/.venv/bin/python}"
VS_PY="${CHTR_VS_PY:-$ROOT_DIR/.venvs/virtual-sketching/bin/python}"

if [[ ! -x "$PIPELINE_PY" ]]; then
  echo "找不到 StrokeExtraction pipeline Python: $PIPELINE_PY" >&2
  exit 2
fi
if [[ ! -x "$VS_PY" ]]; then
  echo "找不到 Virtual Sketching TF1 Python: $VS_PY" >&2
  exit 2
fi

exec "$PIPELINE_PY" "$ROOT_DIR/CHTR/data/build_chtr_dataset.py" \
  --pipeline-python "$PIPELINE_PY" \
  --stroke-python "$PIPELINE_PY" \
  --vs-python "$VS_PY" \
  "$@"
