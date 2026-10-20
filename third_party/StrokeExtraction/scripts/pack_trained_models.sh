#!/usr/bin/env bash
# Package the three trained checkpoints in the layout expected by inference.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-RHSEDB}"
SDNET="$PROJECT_ROOT/out/SDNet_${DATASET}/model/sdnet_model.pth"
SEGNET="$PROJECT_ROOT/out/SegNet_${DATASET}/model/model.pth"
EXTRACTNET="$PROJECT_ROOT/out/ExtractNet_${DATASET}/model/model_extract.pth"

missing=0
for checkpoint in "$SDNET" "$SEGNET" "$EXTRACTNET"; do
    if [[ ! -f "$checkpoint" ]]; then
        printf 'Missing checkpoint: %s\n' "$checkpoint" >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    printf 'Complete all three training stages before packaging.\n' >&2
    exit 1
fi

mkdir -p "$PROJECT_ROOT/dist"
STAGING="$(mktemp -d)"
trap 'rm -rf -- "$STAGING"' EXIT
mkdir -p "$STAGING/model"
cp "$SDNET" "$STAGING/model/sdnet_model.pth"
cp "$SEGNET" "$STAGING/model/model.pth"
cp "$EXTRACTNET" "$STAGING/model/model_extract.pth"

ARCHIVE="$PROJECT_ROOT/dist/StrokeExtraction-models-${DATASET}-$(date +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$ARCHIVE" -C "$STAGING" model
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
printf '%s\n' "$ARCHIVE"
