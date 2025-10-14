#!/bin/bash
#SBATCH --job-name=hindi_babylm_pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@tum.de

# Hindi BabyLM Complete Pipeline on LRZ
# This script runs the complete pipeline: data processing -> training -> evaluation

echo "=========================================="
echo "Hindi BabyLM Complete Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "=========================================="

# Load required modules
module load python/3.10
module load cuda/11.8
module load cudnn/8.6

# Activate virtual environment
source venv/bin/activate

# Print GPU information
echo ""
echo "GPU Information:"
nvidia-smi

# Print Python and PyTorch versions
echo ""
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

# Run the complete pipeline
echo ""
echo "=========================================="
echo "Starting Hindi BabyLM Pipeline"
echo "=========================================="

python main.py \
    --config configs/base_config.yaml \
    --experiment_name ${SLURM_JOB_ID}_complete_pipeline \
    --stage all \
    --device cuda

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Pipeline completed successfully!"
    echo "Results saved to: results/${SLURM_JOB_ID}_complete_pipeline/"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Pipeline failed with error!"
    echo "Check error log: logs/pipeline_${SLURM_JOB_ID}.err"
    echo "=========================================="
    exit 1
fi

echo ""
echo "Finished at: $(date)"
