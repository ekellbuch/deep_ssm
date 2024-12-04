# Deep sequence models

## Installation:

Run the following commands to install the required packages:
```
./setup_env.sh
```
Note: if working on sherlock make sure to download the correct modules before running the above command.

```
ml python/3.9.0 && ml gcc/10.1.0 && ml cudnn/8.9.0.131  && ml load cuda/12.4.1
```

## Installation that worked for XG

Here is a description of the installation procedure that worked for me. 

1. I use python 3.12 in Sherlock. I do this with `module load python/3.12.1`
2. I set up miniconda. Here are some instructions that I have found helpful about setting up miniconda on Sherlock

I then set up a miniconda version of python, following the instructions in the lab manual. Note that I requested a compute node in order to actually install miniconda.

mkdir -p "$SCRATCH"/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$SCRATCH"/miniconda3/miniconda.sh
bash "$SCRATCH"/miniconda3/miniconda.sh -b -u -p "$SCRATCH"/miniconda3
rm -rf "$SCRATCH"/miniconda3/miniconda.sh
"$SCRATCH"/miniconda3/bin/conda init
reload the shell just to double check:

which conda  # Expect: /scratch/users/<sunetid>/miniconda3/bin/conda
Next, we need to ensure that environment paths are prepended to $PATH; this is not automatically done on Linux systems (see issue). To do this, edit your ~/.bashrc file. At the bottom of the file, you will see a code block starting with # >>> conda initialize >>> (this was added when you ran conda init above). Add conda deactivate after this block. Now, restart your shell by running source ~/.bashrc. 

3. `conda create -n nonlinear_ssm`
4. `conda activate nonlinear_ssm`
5. `ml cuda/12.4`
6. `conda install pytorch torchvision torchaudio torchtext pytorch-cuda=12.4 -c pytorch -c nvidia`

7. The two painful installs are `causal-conv1d` and `mamba`. I followed good advice from Noah and Svea, which was basically to git clone them directly. Here are the instructions
```
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.1.1 # this is the lowest compatible version allowed by Mamba
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
cd ..
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v2.2.2
MAMBA_FORCE_BUILD=TRUE pip install .
```
For the rest of these instructions, make sure you are in this `deep_ssm` repo.

8. `pip install --no-cache-dir -r requirements.txt`
9. `pip install -e src/`

At this point, you should be able to go into `scripts` and execute `python run.py` (there will be a lot of text, but eventaully you should see the progress bar from tqdm!


## Baselines:

- Train [S5 model](https://github.com/lindermanlab/s5) on sequential CIFAR10
```
python -m example --grayscale
```

- Train a GRU model on the Brain Computer Interface (BCI) dataset from [Willett et al. 2023](https://github.com/fwillett/speechBCI).
```
1. Download data in directory and export directory path to the environment variable $DEEP_SSM_DATA
gsutil cp gs://cfan/interspeech24/brain2text_competition_data.pkl .
export DEEP_SSM_DATA=/path/to/data

Note: on sherlock the data is already available in the following directory:
export DEEP_SSM_DATA=/scratch/groups/swl1

2. Run code to debug model: 
python run.py --config-name="baseline_gru" trainer_cfg.fast_dev_run=1

3. Full code to train mode: python run.py --config-name="baseline_gru"
python run.py --config-name="baseline_gru"
```


- Train a [Mamba model](https://github.com/state-spaces/mamba) on the Brain Computer Interface (BCI) dataset from [Willett et al. 2023](https://github.com/fwillett/speechBCI)
```
python run.py --config-name="baseline_mamba"
```


