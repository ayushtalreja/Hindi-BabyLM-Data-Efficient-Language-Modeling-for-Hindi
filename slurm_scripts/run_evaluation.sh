#!/bin/bash
#SBATCH --job-name=hindi_evaluation
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=14:00:00
#SBATCH --partition=lrz-dgx-a100-80x8   # Adjust partition as needed   
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayush1.kumar@tum.de

# Hindi BabyLM Evaluation (GPU recommended)
# Evaluates trained model on IndicGLUE, MultiBLiMP, and Morphological Probes

echo "=========================================="
echo "Hindi BabyLM Evaluation"
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

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

# Run evaluation stage
echo ""
echo "=========================================="
echo "Starting Evaluation"
echo "=========================================="

# Parse arguments
CONFIG=${1:-configs/deberta_10M_config.yaml}
EXPERIMENT_NAME=${2:-}  # Optional experiment name override

echo "Using config: $CONFIG"

# Build command
if [ -z "$EXPERIMENT_NAME" ]; then
    # Use experiment name from config file
    echo "Using experiment name from config file"
    python main.py \
        --config $CONFIG \
        --stage eval \
        --device cuda
else
    # Override with provided experiment name
    echo "Using experiment name: $EXPERIMENT_NAME"
    python main.py \
        --config $CONFIG \
        --experiment_name $EXPERIMENT_NAME \
        --stage eval \
        --device cuda
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "Check results directory for evaluation output"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Evaluation failed with error!"
    echo "=========================================="
    exit 1
fi

echo ""
echo "Finished at: $(date)"
