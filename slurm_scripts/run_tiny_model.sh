#!/bin/bash
#SBATCH --job-name=hindi_tiny_model
#SBATCH --output=logs/tiny_%j.out
#SBATCH --error=logs/tiny_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@tum.de

# Hindi BabyLM Tiny Model (50M parameters)
# Quick training for testing and development

echo "=========================================="
echo "Hindi BabyLM Tiny Model Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "=========================================="

# Load modules
module load python/3.10
module load cuda/11.8
module load cudnn/8.6

# Activate environment
source venv/bin/activate

# GPU info
echo ""
nvidia-smi

# Set environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

# Run pipeline with tiny model config
python main.py \
    --config configs/tiny_model.yaml \
    --experiment_name ${SLURM_JOB_ID}_tiny_50m \
    --stage all \
    --device cuda

echo ""
echo "Finished at: $(date)"
