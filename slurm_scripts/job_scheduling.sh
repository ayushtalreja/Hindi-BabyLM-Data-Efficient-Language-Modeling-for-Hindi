#!/bin/bash
#SBATCH --job-name=baseline_experiment
#SBATCH --output=../results/output_%j.log
#SBATCH --partition=lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=4             # Adjust as needed
#SBATCH --mem=32G                     # Adjust as needed
#SBATCH --time=12:00:00               # Job time limit (adjust)

# Load your modules or activate environment
if command -v conda >/dev/null 2>&1; then
    # Prefer activating by name if conda is available
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate hindi_babylm || echo "Warning: couldn't activate conda env 'hindi_babylm'"
fi

# Move to repository root (assumes script is in slurm_scripts/)
cd "$(dirname "$0")/.." || exit 1

# Run your command
python3 main.py \
    --config configs/base_config.yaml \
    --experiment_name baseline_experiment
