#!/usr/bin/bash
#SBATCH --job-name=bci_sweep
#SBATCH --error=bci_sweep_%j_%a.err
#SBATCH --out=bci_sweep_%j_%a.out
#SBATCH --time=6-23:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --constraint='GPU_SKU:A100_SXM4&GPU_MEM:80GB'
#SBATCH --mail-type=ALL

export DEEP_SSM_DATA=/scratch/groups/swl1
module load cuda/12.4


source /scratch/users/xavier18/miniconda3/bin/activate deep_ssm

# Debugging
env | grep PATH
which ninja
python -m pip show ninja


/scratch/users/xavier18/miniconda3/envs/deep_ssm/bin/python3 run.py -m $@