#!/usr/bin/env bash
# Code + lockfile package. Large datasets, model weights and local environments
# are transferred separately; .venv contains absolute interpreter paths.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
test -f "$PROJECT_ROOT/uv.lock" || { echo 'uv.lock missing: run uv sync first.' >&2; exit 1; }
mkdir -p "$PROJECT_ROOT/dist"
ARCHIVE="$PROJECT_ROOT/dist/StrokeExtraction-server-$(date +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$ARCHIVE" \
    --exclude='StrokeExtraction/.git' \
    --exclude='StrokeExtraction/.venv' \
    --exclude='StrokeExtraction/.python' \
    --exclude='StrokeExtraction/.tools' \
    --exclude='StrokeExtraction/.uv-cache' \
    --exclude='StrokeExtraction/dist' \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' --exclude='*.pth' --exclude='*.th' \
    --exclude='*.zip' --exclude='*.npz' \
    --exclude='StrokeExtraction/out' \
    --exclude='StrokeExtraction/dataset_forSegNet_ExtractNet_*' \
    -C "$(dirname "$PROJECT_ROOT")" "$(basename "$PROJECT_ROOT")"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
printf '%s\n' "$ARCHIVE"
