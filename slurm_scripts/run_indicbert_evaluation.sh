#!/bin/bash
#SBATCH --job-name=indicbert_eval
#SBATCH --output=logs/indicbert_%j.out
#SBATCH --error=logs/indicbert_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=lrz-hgx-h100-94x4 #lrz-dgx-a100-80x8   # Adjust partition as needed
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayush1.kumar@tum.de

# IndicBERT Evaluation on IndicGLUE Tasks
# Evaluates pre-trained IndicBERT model from HuggingFace to verify
# the correctness of the existing IndicGLUE evaluation implementation

echo "=========================================="
echo "IndicBERT IndicGLUE Evaluation"
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

# Parse command-line arguments with defaults
MODE=${1:-fine-tune}           # fine-tune or zero-shot
TASKS=${2:-all}                 # all, or space-separated list like "WSTP COPA BBCA"
MAX_SAMPLES=${3:-}              # Optional: limit samples per task for testing
EPOCHS=${4:-10}                 # Number of fine-tuning epochs
BATCH_SIZE=${5:-32}             # Batch size
LEARNING_RATE=${6:-2e-5}        # Learning rate

echo ""
echo "=========================================="
echo "Configuration"
echo "=========================================="
echo "Mode: $MODE"
echo "Tasks: $TASKS"
echo "Max samples per task: ${MAX_SAMPLES:-all}"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo ""

# Build command
CMD="python scripts/evaluate_indicbert.py \
    --mode $MODE \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --device cuda \
    --output-dir results/indicbert_evaluation_${SLURM_JOB_ID}"

# Add tasks if specified
if [ "$TASKS" != "all" ]; then
    CMD="$CMD --tasks $TASKS"
fi

# Add max samples if specified (for quick testing)
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
    echo "Running in TEST MODE with $MAX_SAMPLES samples per task"
fi

# Run evaluation
echo ""
echo "=========================================="
echo "Starting IndicBERT Evaluation"
echo "=========================================="
echo "Command: $CMD"
echo ""

eval $CMD

# Check exit status
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "=========================================="
    echo "Results saved to: results/indicbert_evaluation_${SLURM_JOB_ID}/"
    echo ""
    echo "Output files:"
    echo "  - indicbert_results_summary.csv (for comparison with paper)"
    echo "  - indicbert_results_detailed.json (complete metrics)"
    echo "  - indicbert_evaluation.log (execution log)"
    echo ""
    echo "Next steps:"
    echo "  1. Check summary CSV: cat results/indicbert_evaluation_${SLURM_JOB_ID}/indicbert_results_summary.csv"
    echo "  2. Compare with IndicBERT paper scores (Table 4 for Hindi)"
    echo "  3. If results match → Your IndicGLUE evaluation is verified! ✓"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Evaluation failed with error code: $EXIT_CODE"
    echo "=========================================="
    echo "Check the error log: logs/indicbert_${SLURM_JOB_ID}.err"
    echo "Check the execution log: results/indicbert_evaluation_${SLURM_JOB_ID}/indicbert_evaluation.log"
    exit 1
fi

echo ""
echo "Finished at: $(date)"
