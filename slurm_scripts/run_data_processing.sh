#!/bin/bash
#SBATCH --job-name=hindi_data_processing
#SBATCH --output=logs/data_%j.out
#SBATCH --error=logs/data_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=lrz-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayush1.kumar@tum.de

# Hindi BabyLM Data Processing (CPU only - no GPU needed)
# Downloads and processes data from IndicCorp, Wikipedia, and children's books

echo "=========================================="
echo "Hindi BabyLM Data Processing"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "=========================================="

# Load required modules
module load python/3.10

# Activate virtual environment
source venv/bin/activate

# Print Python version
echo ""
echo "Python version:"
python --version

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run data processing stage
echo ""
echo "=========================================="
echo "Starting Data Processing"
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
        --stage data \
        --force-reprocess
else
    # Override with provided experiment name
    echo "Using experiment name: $EXPERIMENT_NAME"
    python main.py \
        --config $CONFIG \
        --experiment_name $EXPERIMENT_NAME \
        --stage data \
        --force-reprocess
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Data processing completed successfully!"
    echo "Check data directory for splits"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Data processing failed with error!"
    echo "=========================================="
    exit 1
fi

echo ""
echo "Finished at: $(date)"
