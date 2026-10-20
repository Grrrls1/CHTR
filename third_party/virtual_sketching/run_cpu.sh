#!/usr/bin/env bash
set -euo pipefail
VS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CHTR_DIR="$(cd -- "$VS_ROOT/../.." && pwd -P)"
VS_DEFAULT_PY="$CHTR_DIR/.venvs/virtual-sketching/bin/python"
if [[ ! -x "$VS_DEFAULT_PY" ]]; then
  VS_DEFAULT_PY="$CHTR_DIR/../.venvs/virtual-sketching/bin/python"
fi
VS_PYTHON="${VS_PYTHON:-$VS_DEFAULT_PY}"
cd "$VS_ROOT"
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
export PYTHONPATH=
export OMP_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2
# This environment installs the CPU-only TensorFlow 1.15 wheel.
exec "$VS_PYTHON" test_vectorization.py "$@"
