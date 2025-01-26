#!/usr/bin/bash
#SBATCH --job-name='slurm_ekb'
#SBATCH --output=slurm_ekb-%j.out
#SBATCH --time=05:00:00  #=<TOTAL_TIME_PER_JOB>
#SBATCH -G 1   # number of gpus
#SBATCH --cpus-per-gpu 16  # number of cpus
#SBATCH -p swl1,owners,stat # partition
#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ekb@stanford.edu
#SBATCH --mem=16G

# make cuda run asynchronous operations
# export CUDA_LAUNCH_BLOCKING=1

# Nice print statements:
echo SLURM_JOB_ID $SLURM_JOB_ID

# Load some modules
ml python/3.9.0
ml gcc/10.1.0
ml cudnn/8.9.0.131
ml cuda/12.4.0

# Activate virtual environment (optional but you'll probably need it)
#conda init
eval "$(conda shell.bash hook)"
conda activate deep_ssm

# code to run
python run_experiment.py -m SequenceModel -n Mamba -l 16 -e A5 -d 4
python run_experiment.py -m SequenceModel -n MambaAComplex -l 16 -e A5 -d 4
python run_experiment.py -m SequenceModel -n MambaABCComplex -l 16 -e A5 -d 4

echo "Ran $SLURM_ARRAY_JOB_ID$"

deactivate
