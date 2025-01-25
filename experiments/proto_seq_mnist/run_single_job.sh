#!/bin/bash
#SBATCH --job-name=single_job
#SBATCH --error=.logs/single_job_%j_%a.err         
#SBATCH --output=.logs/single_job_%j_%a.out     
#SBATCH --time=03:59:59                 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mail-type=ALL
#SBATCH --requeue
#SBATCH --constraint='GPU_MEM:48GB|GPU_MEM:80GB'

# Environment setup
module load cuda/12.4

# Activate conda environment
source /scratch/users/xavier18/miniconda3/bin/activate nonlinear_ssm

# Ensure the .logs/ directory exists
LOG_DIR=".logs"
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

python seq_mnist_hydra.py $@