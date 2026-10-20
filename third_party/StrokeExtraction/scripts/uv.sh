#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$PROJECT_ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.python"
cd "$PROJECT_ROOT"
if [[ -x "$PROJECT_ROOT/.tools/bin/uv" ]]; then
    exec "$PROJECT_ROOT/.tools/bin/uv" "$@"
fi
exec uv "$@"
