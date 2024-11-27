#!/usr/bin/bash
#SBATCH --job-name=bci
#SBATCH --error=bci_%j_%a.err
#SBATCH --out=bci_%j_%a.out
#SBATCH --time=04:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --constraint='GPU_SKU:A100_SXM4&GPU_MEM:80GB'
#SBATCH --mail-type=ALL

export DEEP_SSM_DATA=/scratch/groups/swl1
module load cuda/12.4


eval "$(conda shell.bash hook)"
conda activate deep_ssm

# Debugging
env | grep PATH
which ninja
python -m pip show ninja


/scratch/users/xavier18/miniconda3/envs/deep_ssm/bin/python3 run.py $@