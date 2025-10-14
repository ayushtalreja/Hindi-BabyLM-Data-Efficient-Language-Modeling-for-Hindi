# LRZ Quick Reference Card

## Essential Commands

### Job Submission
```bash
# Complete pipeline
sbatch slurm_scripts/run_complete_pipeline.sh

# Data processing only
sbatch slurm_scripts/run_data_processing.sh

# Training only
sbatch slurm_scripts/run_training.sh

# Evaluation only
sbatch slurm_scripts/run_evaluation.sh <experiment_name>

# Quick test (Tiny model)
sbatch slurm_scripts/run_tiny_model.sh
```

### Job Monitoring
```bash
# Check your jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# Watch jobs (auto-refresh every 30s)
watch -n 30 'squeue -u $USER'

# Cancel job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

### Log Viewing
```bash
# Real-time output
tail -f logs/pipeline_<job_id>.out

# Real-time errors
tail -f logs/pipeline_<job_id>.err

# View last 100 lines
tail -n 100 logs/pipeline_<job_id>.out

# Search logs
grep "ERROR" logs/pipeline_<job_id>.err
```

### Results
```bash
# List experiments
ls -lh results/

# View evaluation results
cat results/<experiment_name>/evaluation_results.json | jq .

# View training summary
cat results/<experiment_name>/training_summary.json | jq .

# Check if experiment completed
ls results/<experiment_name>/COMPLETED
```

## Common Scenarios

### Scenario 1: First Time Running
```bash
# Setup
module load python/3.10
source venv/bin/activate
mkdir -p logs

# Update email
nano slurm_scripts/run_complete_pipeline.sh
# Change: your.email@tum.de → YOUR_EMAIL@tum.de

# Submit
sbatch slurm_scripts/run_complete_pipeline.sh
```

### Scenario 2: Resume Failed Training
```bash
# Find last checkpoint
ls results/<experiment_name>/checkpoints/

# Edit training script to add resume
nano slurm_scripts/run_training.sh
# Add: --resume results/<experiment_name>/checkpoints/checkpoint_epoch_N.pt

# Resubmit
sbatch slurm_scripts/run_training.sh
```

### Scenario 3: Run Multiple Experiments
```bash
# Create job array script
nano slurm_scripts/run_experiments_array.sh

# Add to script:
#SBATCH --array=1-5
configs=("configs/tiny.yaml" "configs/small.yaml" "configs/medium.yaml" "configs/curriculum.yaml" "configs/rope.yaml")
python main.py --config ${configs[$SLURM_ARRAY_TASK_ID-1]} --experiment_name exp_$SLURM_ARRAY_TASK_ID

# Submit array
sbatch slurm_scripts/run_experiments_array.sh
```

### Scenario 4: Out of Memory
```bash
# Option 1: Reduce batch size
nano configs/base_config.yaml
# Change: batch_size: 32 → batch_size: 16

# Option 2: Request more memory
nano slurm_scripts/run_training.sh
# Change: #SBATCH --mem=64GB → #SBATCH --mem=128GB

# Option 3: Use tiny model
sbatch slurm_scripts/run_tiny_model.sh
```

### Scenario 5: Job Time Limit
```bash
# Request more time
nano slurm_scripts/run_training.sh
# Change: #SBATCH --time=48:00:00 → #SBATCH --time=96:00:00

# Or enable checkpointing and resume
# Config ensures checkpoints saved every epoch
```

## Module Loading
```bash
# Standard modules for this project
module load python/3.10
module load cuda/11.8
module load cudnn/8.6

# List loaded modules
module list

# Purge all modules
module purge

# Search for modules
module avail cuda
```

## File Transfers

### From Local to LRZ
```bash
# Single file
scp file.txt <username>@login.lrz.de:~/hindi-babylm/

# Directory
scp -r configs/ <username>@login.lrz.de:~/hindi-babylm/

# Using rsync (recommended)
rsync -avz --progress configs/ <username>@login.lrz.de:~/hindi-babylm/configs/
```

### From LRZ to Local
```bash
# Results directory
scp -r <username>@login.lrz.de:~/hindi-babylm/results/12345678_experiment ./

# Specific files
scp <username>@login.lrz.de:~/hindi-babylm/results/*/evaluation_results.json ./

# Using rsync
rsync -avz --progress <username>@login.lrz.de:~/hindi-babylm/results/ ./results/
```

## Disk Usage
```bash
# Check your quota
lfs quota -h $HOME

# Check directory size
du -sh results/
du -sh data/

# Find large files
find results/ -type f -size +1G -exec ls -lh {} \;

# Clean up old checkpoints
find results/*/checkpoints/ -name "checkpoint_epoch_*.pt" | sort -V | head -n -3 | xargs rm
```

## GPU Information
```bash
# Check available GPUs in partition
sinfo -p gpu_4

# During job execution, SSH to node and check
ssh <node_name>
nvidia-smi
nvidia-smi -l 5  # Update every 5 seconds
```

## Environment Variables
```bash
# Useful SLURM variables in scripts
echo $SLURM_JOB_ID       # Job ID
echo $SLURM_JOB_NAME     # Job name
echo $SLURM_NODELIST     # Allocated nodes
echo $SLURM_CPUS_PER_TASK # CPUs allocated

# Set in scripts for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0
```

## Troubleshooting Quick Fixes

### Python Module Not Found
```bash
source venv/bin/activate
pip install -e .
```

### CUDA Not Available
```bash
module load cuda/11.8
module load cudnn/8.6
python -c "import torch; print(torch.cuda.is_available())"
```

### Permission Denied
```bash
chmod +x slurm_scripts/*.sh
```

### Disk Quota Exceeded
```bash
# Remove old logs
rm logs/pipeline_*.out logs/pipeline_*.err

# Remove old checkpoints (keep only best)
cd results/<experiment>/checkpoints/
ls -t checkpoint_epoch_*.pt | tail -n +4 | xargs rm
```

## Priority and Partitions

### LRZ GPU Partitions
- **gpu_4**: A100 GPUs (recommended) - 48h max
- **gpu_8**: V100 GPUs - 48h max
- **serial_std**: CPU only - for data processing

### Check Partition Status
```bash
sinfo -p gpu_4
scontrol show partition gpu_4
```

## Contact & Help

- **LRZ Service Desk**: servicedesk@lrz.de
- **LRZ Documentation**: https://doku.lrz.de/
- **Project Issues**: GitHub Issues

## Useful Aliases

Add to your `~/.bashrc`:

```bash
alias sq='squeue -u $USER'
alias sc='scancel'
alias si='sinfo -p gpu_4'
alias ll='ls -lhrt'
alias ta='tail -f logs/pipeline_*.out'
alias te='tail -f logs/pipeline_*.err'
alias res='cd ~/hindi-babylm/results && ls -lhrt'
alias act='source ~/hindi-babylm/venv/bin/activate'
```

Then: `source ~/.bashrc`
