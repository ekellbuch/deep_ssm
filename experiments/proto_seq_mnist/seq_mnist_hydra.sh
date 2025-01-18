#!/bin/bash
#SBATCH --job-name=smnist 
#SBATCH --error=.logs/smnist_%j_%a.err         
#SBATCH --output=.logs/smnist_%j_%a.out     
#SBATCH --time=47:59:59                 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --constraint='GPU_SKU:A100_SXM4&GPU_MEM:80GB'
#SBATCH --mail-type=ALL
#SBATCH --requeue

# Environment setup
module load cuda/12.4

# Activate conda environment
source /scratch/users/xavier18/miniconda3/bin/activate nonlinear_ssm

# Ensure the .logs/ directory exists
LOG_DIR=".logs"
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

# Get Sweep ID from the command line argument
SWEEP_ID=$1

# Check if a Sweep ID was provided
if [ -z "$SWEEP_ID" ]; then
  echo "Error: No Sweep ID provided. Usage: sbatch wandb_agent.sbatch <SWEEP_ID>"
  exit 1
fi

# Construct the full Sweep Identifier
FULL_SWEEP_ID="xavier_gonzalez/nonlinear_ssm/$SWEEP_ID"

# logic to make robust to preemption
# Define checkpoint file
CHECKPOINT_FILE=".checkpoint/${SLURM_JOB_ID}.ckpt"
mkdir -p .checkpoint

# Handle preemption (trap SIGTERM)
trap 'wandb agent --resume $FULL_SWEEP_ID' SIGTERM

# Run the W&B agent
if [ -f "$CHECKPOINT_FILE" ]; then
  echo "Resuming from checkpoint..."
  wandb agent --resume $FULL_SWEEP_ID
else
  echo "Starting new W&B run..."
  wandb agent $FULL_SWEEP_ID
fi