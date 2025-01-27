#!/usr/bin/bash
#SBATCH --job-name='slurm_ekb'
#SBATCH --output=slurm_ekb-%A_%a.out
#SBATCH --time=06:00:00  #=<TOTAL_TIME_PER_JOB>
#SBATCH -G 1   # number of gpus
#SBATCH --cpus-per-gpu 16  # number of cpus
#SBATCH -p gpu,swl1,owners,stat  # partition
#SBATCH --mail-type=FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ekb@stanford.edu
#SBATCH --array=1-4
#SBATCH --mem=32G

# Check how many jobs i can submit:
# sacctmgr show qos format=Name,MaxSubmitJobsPerUser,MaxJobsPerUser
# check usage: sh_part
# check priority sprio -j 56511670
# check job priority:
# scontrol show job 56511670
# check gpu usage:
#  node_feat -p gpu | grep GPU_


# Nice print statements:
# Print the array task ID
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
# Example array of jobs
N_CONFIGS_PER_AGENT=1
export SWEEP_ID=nzm6ix5h
export WANDB_ENTITY=hazy-research
export WANDB_PROJECT=complex_mamba_group_multiplication

# Nice print statements (optional)
echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID
echo SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID
echo SLURM_JOB_ID $SLURM_JOB_ID
echo SWEEP_ID $SWEEP_ID
# Load some modules
module load python/3.9.0
ml gcc/10.1.0
ml load cudnn/8.9.0.131
ml load cuda/12.4.0  # could be 11.7.1
#your config file specifies a total of N_CONFIGS
# N_CONFIGS_PER_AGENT x N_SLURM_JOBS = N_CONFIGS
# N_CONFIGS_PER_AGENT x TIME_PER_RUN = TOTAL_TIME_PER_SLURM_JOB x N_SLURM_JOBS (add some percentage buffer room on TOTAL_TIME_PER_SLURM_JOB )
# Choose N_CONFIGS_PER_AGENT, TOTAL_TIME_PER_SLURM_JOB, and N_SLURM_JOBS (there is some limit, but I can't remember) to get "best" slurm job score for you
# Activate virtual environment (optional but you'll probably need it)
eval "$(conda shell.bash hook)"
conda activate deep_ssm
# Technically should not need to specify entity and project, but I've found that I hit bugs otherwise
wandb agent --count $N_CONFIGS_PER_AGENT "$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"
echo "Ran $SLURM_ARRAY_TASK_ID$"
deactivate
