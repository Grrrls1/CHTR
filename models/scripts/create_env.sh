#!/usr/bin/env bash
# Create an isolated environment on the training machine, never the global Python.
set -euo pipefail
model_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python=python3
env_dir="$model_root/.venv"
backend=cu124
create_only=false
usage() {
  cat <<'HELP'
Usage: bash create_env.sh [--python python3.10] [--env-dir PATH]
                          [--backend cu118|cu121|cu124|cpu] [--create-only]
Default: .venv under models, PyTorch 2.5.1 + torchvision 0.20.1, CUDA 12.4.
--create-only creates the environment without downloading dependencies.
HELP
}
while (($#)); do
  case "$1" in
    --python|--env-dir|--backend)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --python) task_python="$2" ;;
        --env-dir) env_dir="$2" ;;
        --backend) backend="$2" ;;
      esac
      shift 2 ;;
    --create-only) create_only=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
case "$backend" in cu118|cu121|cu124|cpu) ;; *) echo "Unsupported backend: $backend" >&2; exit 2 ;; esac
"$task_python" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10, 3.11 or 3.12 for this pinned environment"'
if [[ -e "$env_dir" && ! -f "$env_dir/pyvenv.cfg" ]]; then
  echo "Destination exists but is not a venv: $env_dir" >&2; exit 2
fi
if [[ ! -f "$env_dir/pyvenv.cfg" ]]; then
  "$task_python" -m venv "$env_dir"
fi
env_python="$env_dir/bin/python"
"$env_python" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12)'
if [[ "$create_only" == true ]]; then
  echo "Environment created: $env_dir (dependencies not installed)"; exit 0
fi
"$env_python" -m pip install --upgrade pip
"$env_python" -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url "https://download.pytorch.org/whl/$backend"
"$env_python" -m pip install -r "$model_root/requirements-env.txt"
"$env_python" -m pip check
"$env_python" "$model_root/scripts/check_environment.py" --require-torchvision
printf 'Environment ready: %s\nActivate with: source "%s/bin/activate"\n' "$env_dir" "$env_dir"
