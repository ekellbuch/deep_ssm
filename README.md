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


