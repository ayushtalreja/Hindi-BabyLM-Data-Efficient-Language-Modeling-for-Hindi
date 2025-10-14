#!/bin/bash
#SBATCH --job-name=hindi_curriculum
#SBATCH --output=logs/curriculum_%j.out
#SBATCH --error=logs/curriculum_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@tum.de

# Hindi BabyLM with Curriculum Learning
# Trains with morphological curriculum strategy

echo "=========================================="
echo "Hindi BabyLM Curriculum Learning"
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

# Run with curriculum learning config
python main.py \
    --config configs/curriculum_learning.yaml \
    --experiment_name ${SLURM_JOB_ID}_curriculum \
    --stage all \
    --device cuda

echo ""
echo "Finished at: $(date)"
