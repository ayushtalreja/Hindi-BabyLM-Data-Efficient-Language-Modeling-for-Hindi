# Running Hindi BabyLM on LRZ Systems

This guide explains how to run the Hindi BabyLM pipeline on LRZ (Leibniz Rechenzentrum) GPU clusters.

## Prerequisites

### 1. LRZ Account Setup

Ensure you have:
- Active LRZ account
- Access to GPU partitions (apply via LRZ service desk if needed)
- SSH access configured

### 2. Initial Setup on LRZ

```bash
# Login to LRZ
ssh <username>@login.lrz.de

# Navigate to your workspace
cd $HOME/workspace  # or your preferred directory

# Clone the repository
git clone https://github.com/yourusername/hindi-babylm.git
cd hindi-babylm

# Create logs directory for SLURM output
mkdir -p logs

# Load Python module
module load python/3.10

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

### 3. Update Email Address

Edit SLURM scripts to use your TUM email:

```bash
# Update email in all scripts
sed -i 's/your.email@tum.de/your.actual.email@tum.de/g' slurm_scripts/*.sh
```

## Available SLURM Scripts

| Script | Purpose | Time | GPU | Memory | Partition |
|--------|---------|------|-----|--------|-----------|
| `run_complete_pipeline.sh` | Complete pipeline | 24h | 1 | 64GB | gpu_4 |
| `run_data_processing.sh` | Data only (CPU) | 4h | 0 | 32GB | serial_std |
| `run_training.sh` | Training only | 48h | 1 | 64GB | gpu_4 |
| `run_evaluation.sh` | Evaluation only | 8h | 1 | 32GB | gpu_4 |
| `run_tiny_model.sh` | Quick test (50M) | 12h | 1 | 32GB | gpu_4 |
| `run_curriculum_learning.sh` | Curriculum learning | 48h | 1 | 64GB | gpu_4 |

## Usage

### Option 1: Complete Pipeline (Recommended for First Run)

Run everything in one job:

```bash
sbatch slurm_scripts/run_complete_pipeline.sh
```

This will:
1. Process data from IndicCorp, Wikipedia, children's books
2. Create train/val/test splits
3. Train the language model
4. Run comprehensive evaluation
5. Save all results to `results/[job_id]_complete_pipeline/`

### Option 2: Stage-by-Stage Execution

#### Step 1: Data Processing (CPU only)

```bash
sbatch slurm_scripts/run_data_processing.sh
```

This downloads and processes all data, saving splits to `data/splits/`.

#### Step 2: Training (GPU required)

```bash
# Basic training with base config
sbatch slurm_scripts/run_training.sh

# With custom config
sbatch slurm_scripts/run_training.sh configs/small_model.yaml my_experiment

# With curriculum learning
sbatch slurm_scripts/run_curriculum_learning.sh
```

#### Step 3: Evaluation (GPU recommended)

```bash
sbatch slurm_scripts/run_evaluation.sh <experiment_name>

# Example:
sbatch slurm_scripts/run_evaluation.sh 12345678_training
```

### Option 3: Quick Testing with Tiny Model

For rapid prototyping and testing:

```bash
sbatch slurm_scripts/run_tiny_model.sh
```

Trains a 50M parameter model (much faster than 110M or 350M).

## Monitoring Jobs

### Check Job Status

```bash
# List your jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# Cancel a job
scancel <job_id>
```

### View Logs in Real-Time

```bash
# Standard output
tail -f logs/pipeline_<job_id>.out

# Error output
tail -f logs/pipeline_<job_id>.err
```

### Check GPU Usage

```bash
# If your job is running
ssh <compute_node>  # Get node name from squeue
nvidia-smi
```

## LRZ-Specific Configurations

### GPU Partitions

LRZ provides several GPU partitions:

- **`gpu_4`**: NVIDIA A100 GPUs (recommended for this project)
  - 4 GPUs per node
  - Up to 48 hours walltime
  - 64-512GB RAM per node

- **`gpu_8`**: NVIDIA V100 GPUs
  - 8 GPUs per node
  - Up to 48 hours walltime

### Requesting More Resources

Edit the SLURM script headers:

```bash
#SBATCH --time=72:00:00        # Increase time
#SBATCH --mem=128GB            # Increase memory
#SBATCH --cpus-per-task=16     # More CPUs
```

### Using Multiple GPUs (Advanced)

For multi-GPU training, modify the script:

```bash
#SBATCH --gres=gpu:2           # Request 2 GPUs
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
```

Then update your config to enable distributed training.

## Troubleshooting

### Issue: Module Not Found

```bash
# Reload modules
module purge
module load python/3.10
module load cuda/11.8
module load cudnn/8.6
source venv/bin/activate
```

### Issue: Out of Memory (OOM)

**Solutions:**
1. Reduce batch size in config file
2. Use smaller model (tiny instead of small/medium)
3. Request more memory in SLURM script
4. Enable gradient checkpointing (if implemented)

### Issue: CUDA Out of Memory

**Solutions:**
1. Reduce `batch_size` in config
2. Reduce `max_position_embeddings` (e.g., 512 instead of 1024)
3. Use mixed precision training (`fp16` or `bf16`)
4. Use gradient accumulation

### Issue: Job Time Limit

If your job is killed due to time limit:

1. **Resume training:**
```bash
# Edit training script to add --resume flag
python main.py \
    --config configs/base_config.yaml \
    --stage train \
    --resume results/<experiment>/checkpoints/checkpoint_epoch_N.pt
```

2. **Request more time:**
```bash
#SBATCH --time=96:00:00  # 4 days
```

### Issue: Data Download Fails

If IndicCorp download fails due to network issues:

```bash
# Pre-download on login node (has better internet)
python src/data_processing/indiccorp_downloader.py \
    --output-dir data/raw \
    --num-samples 100000 \
    --format both

# Then run pipeline with --force-reprocess
```

## Resource Recommendations

### For Development/Testing
- **Model**: Tiny (50M parameters)
- **Time**: 12 hours
- **GPU**: 1x A100
- **Memory**: 32GB
- **Script**: `run_tiny_model.sh`

### For Research Experiments
- **Model**: Small (110M parameters)
- **Time**: 48 hours
- **GPU**: 1x A100
- **Memory**: 64GB
- **Script**: `run_complete_pipeline.sh`

### For Production/Publication
- **Model**: Medium (350M parameters)
- **Time**: 72-96 hours
- **GPU**: 1x A100
- **Memory**: 128GB
- **Custom Script**: Copy and modify `run_training.sh`

## Best Practices

1. **Test First**: Always run `run_tiny_model.sh` first to verify everything works

2. **Save Checkpoints**: Ensure your config has checkpoint saving enabled:
   ```yaml
   training:
     save_checkpoints: true
     checkpoint_interval: 1  # Save every epoch
   ```

3. **Monitor Early**: Check logs after 10-15 minutes to catch early errors

4. **Use Job Arrays**: For multiple experiments, use SLURM job arrays:
   ```bash
   #SBATCH --array=1-5
   ```

5. **Clean Up**: Remove old checkpoints and logs periodically:
   ```bash
   # Keep only last 3 checkpoints
   find results/*/checkpoints/ -name "checkpoint_epoch_*.pt" | sort -V | head -n -3 | xargs rm
   ```

## Example Workflow

Complete workflow for running experiments:

```bash
# 1. Initial setup (once)
module load python/3.10
source venv/bin/activate

# 2. Process data (once)
sbatch slurm_scripts/run_data_processing.sh

# 3. Wait for data job to complete
squeue -u $USER

# 4. Run baseline experiment
sbatch slurm_scripts/run_complete_pipeline.sh

# 5. Run curriculum learning experiment
sbatch slurm_scripts/run_curriculum_learning.sh

# 6. Monitor jobs
watch -n 30 'squeue -u $USER'

# 7. Once complete, view results
cd results/
ls -lh
cat */evaluation_results.json
```

## Getting Help

- **LRZ Documentation**: https://doku.lrz.de/
- **GPU User Guide**: https://doku.lrz.de/display/PUBLIC/GPU+Usage
- **Project Issues**: https://github.com/yourusername/hindi-babylm/issues
- **LRZ Service Desk**: servicedesk@lrz.de

## Quick Reference Commands

```bash
# Submit job
sbatch slurm_scripts/run_complete_pipeline.sh

# Check queue
squeue -u $USER

# Cancel job
scancel <job_id>

# View output
tail -f logs/pipeline_<job_id>.out

# Check node info
sinfo -p gpu_4

# Check your quota
lfs quota -h /dss/dss...
```

---

**Note**: Replace `<username>`, `<job_id>`, and email addresses with your actual values before running commands.
