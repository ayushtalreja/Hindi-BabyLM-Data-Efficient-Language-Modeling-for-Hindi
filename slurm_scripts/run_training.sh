#!/bin/bash
#SBATCH --job-name=hindi_training
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@tum.de

# Hindi BabyLM Training (GPU required)
# Trains the language model on processed data

echo "=========================================="
echo "Hindi BabyLM Training"
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

# Print versions
echo ""
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

# Run training stage
echo ""
echo "=========================================="
echo "Starting Training"
echo "=========================================="

# Parse arguments - allow config file and experiment name override
CONFIG=${1:-configs/base_config.yaml}
EXPERIMENT_NAME=${2:-}  # Optional experiment name override

echo "Using config: $CONFIG"

# Build command
if [ -z "$EXPERIMENT_NAME" ]; then
    # Use experiment name from config file
    echo "Using experiment name from config file"
    python main.py \
        --config $CONFIG \
        --stage train \
        --device cuda
else
    # Override with provided experiment name
    echo "Using experiment name: $EXPERIMENT_NAME"
    python main.py \
        --config $CONFIG \
        --experiment_name $EXPERIMENT_NAME \
        --stage train \
        --device cuda
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Check results directory for output"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed with error!"
    echo "=========================================="
    exit 1
fi

echo ""
echo "Finished at: $(date)"
