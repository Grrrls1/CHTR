#!/bin/bash
#SBATCH --job-name=chtr-assets
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit with gpu-sbatch --user_name YOUR_NAME}"
package_project="$SLURM_SUBMIT_DIR"
if [[ ! -f "$package_project/reproduction/package_assets.py" ]]; then
  package_project="$package_project/CHTR"
fi
export PYTHONUNBUFFERED=1
srun python3 "$package_project/reproduction/package_assets.py" "$@" \
  --threads "${SLURM_CPUS_PER_TASK:-2}"
