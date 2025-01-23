import os
os.environ['HF_HOME'] = '/scratch/users/ekb/tmp'
os.environ["HF_DATASETS_CACHE"] = '/scratch/users/ekb/tmp'

import hydra
import lightning as L
import wandb

from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from deep_ssm.data.data_loader_state_tracking import GroupMultiplicationDataModule
from deep_ssm.modules import all_modules
from deep_ssm.models import all_models
from omegaconf import OmegaConf
from deep_ssm.utils.callbacks import all_callbacks

import torch
torch.set_float32_matmul_precision('medium')  # Or 'high'
import pandas as pd
import shutil

import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
logger = None
# set callbacks
local_callbacks = []



parser = argparse.ArgumentParser(description="A script that greets the user.")
parser.add_argument("--len", type=int, required=True, help="Length of data.")

# Parse the arguments
argus = parser.parse_args()



BASEDIR="/scratch/groups/swl1/group_multiplication/checkpoints"

L.seed_everything(0)

# For a fixed data k
k = 32
experiment_name = 'mamba_S5_32_0'
args = OmegaConf.load("../configs/group_multiplication/mamba.yaml")
args.data_cfg.k = k
datamodule = GroupMultiplicationDataModule(args.data_cfg)
vocab_size = datamodule.n_vocab

# get module and model:
modelito = all_models[args.model_cfg.type](vocab_size=vocab_size,
                                           **args.model_cfg.configs)
model = all_modules[args.module_cfg.type](args, modelito)

# set trainer:
trainer_config = OmegaConf.to_container(args.trainer_cfg)
trainer_config['logger'] = logger

trainer = L.Trainer(**trainer_config, callbacks=local_callbacks, gradient_clip_val=1.0)
ckpt_path = os.path.join(f"{BASEDIR}/{experiment_name}", "model.ckpt")
mamba_test_acc = trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)[0]['avg_test_acc']
print(f'{k}: {mamba_test_acc}')

# Now load more things

ks = [2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#ks = [10, 30, 40, 20, 50, 60, 70, 80, 90, 100, 8]
num_iters = [4,5,6,7,8,9,10,12,14,16,18,20]
num_eval_iters = [1, 5, 10, 15, 20, 40, 60]
mamba_test_accs = []
sequential_gru_test_accs = []
quasideer_gru_test_accs = defaultdict(list)


# For each Dataset size
#for k in ks:
for k in [argus.len]:
    baseline_results = []
    quasideer_results = []

    # Load mamba:
    L.seed_everything(0)
    experiment_name = 'mamba_S5_32_0'
    args = OmegaConf.load("../configs/group_multiplication/mamba.yaml")
    args.data_cfg.k = k
    datamodule = GroupMultiplicationDataModule(args.data_cfg)
    vocab_size = datamodule.n_vocab

    # get module and model:
    modelito = all_models[args.model_cfg.type](vocab_size=vocab_size,
                                               **args.model_cfg.configs)
    model = all_modules[args.module_cfg.type](args, modelito)

    # set trainer:
    trainer_config = OmegaConf.to_container(args.trainer_cfg)
    trainer_config['logger'] = logger

    trainer = L.Trainer(**trainer_config, callbacks=local_callbacks, gradient_clip_val=1.0)
    ckpt_path = os.path.join(f"{BASEDIR}/{experiment_name}", "model.ckpt")

    mamba_test_acc = trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)[0]['avg_test_acc']
    print(f'{k}: {mamba_test_acc}')
    mamba_test_accs.append(mamba_test_acc)

    # Load sequential
    L.seed_everything(0)
    experiment_name = 'sequential_gru_S5_32_0'
    args = OmegaConf.load("../configs/group_multiplication/sequential_gru_v2.yaml")
    args.data_cfg.k = k
    vocab_size = datamodule.n_vocab

    # get module and model:
    modelito = all_models[args.model_cfg.type](vocab_size=vocab_size,
                                               **args.model_cfg.configs)
    model = all_modules[args.module_cfg.type](args, modelito)

    # set trainer:
    trainer_config = OmegaConf.to_container(args.trainer_cfg)
    trainer_config['logger'] = logger

    trainer = L.Trainer(**trainer_config, callbacks=local_callbacks, gradient_clip_val=1.0)
    ckpt_path = os.path.join(f"{BASEDIR}/{experiment_name}", "model.ckpt")
    try:
        sequential_gru_test_acc = trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)[0]['avg_test_acc']
        print(f'{k}: {sequential_gru_test_acc}')
        sequential_gru_test_accs.append(sequential_gru_test_acc)
    except:
        sequential_gru_test_acc = None
    main_results = {
        "Dataset Length (k)": k,
        "Mamba Test Accuracies": mamba_test_acc,
        "Sequential GRU Test Accuracies": sequential_gru_test_acc,
    }

    baseline_results.append(main_results)

    # Now Load quasi deer results
    # Load for each iteration:
    for num_iter in num_iters:
        L.seed_everything(0)
        experiment_name = f'quasideer_gru_S5_32_gru_{num_iter}_0'
        args = OmegaConf.load("../configs/group_multiplication/quasideer_gru.yaml")
        args.data_cfg.k = k
        args.model_cfg.configs.hidden_size = 370
        args.model_cfg.configs.num_layers = 1
        vocab_size = datamodule.n_vocab

        # Setup different number of iterations
        for num_eval_iter in num_eval_iters:
            L.seed_everything(0)
            wandb.init(mode='disabled')
            args.model_cfg.configs.num_iters = num_eval_iter

            # get module and model:
            modelito = all_models[args.model_cfg.type](vocab_size=vocab_size,
                                                       **args.model_cfg.configs)
            model = all_modules[args.module_cfg.type](args, modelito)

            # set trainer:
            trainer_config = OmegaConf.to_container(args.trainer_cfg)
            trainer_config['logger'] = logger

            trainer = L.Trainer(**trainer_config, callbacks=local_callbacks, gradient_clip_val=1.0)
            ckpt_path = os.path.join(f"{BASEDIR}/{experiment_name}", "model.ckpt")
            try:
                quasideer_gru_test_acc = trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)[0]['avg_test_acc']
                print(f'data length {k}, train iter {num_iter} eval iter {num_eval_iter}: {quasideer_gru_test_acc}')
                quasideer_gru_test_accs[num_iter].append(quasideer_gru_test_acc)
            except:
                quasideer_gru_test_acc = None

            other_results = {
                "Dataset Length (k)": k,
                "QDeer Test Accuracies": quasideer_gru_test_acc,
                "Train iters":num_iter,
                "Eval iters":num_eval_iter,
            }

            quasideer_results.append(other_results)

    # Finally save all results:
    baseline_results_df = pd.DataFrame(baseline_results)
    quasideer_results_df = pd.DataFrame(quasideer_results)

    # Save results to CSV
    print(baseline_results_df)
    print(quasideer_results_df)
    main_results_csv_path = f"main_results_len{k}.csv"
    quasideer_results_csv_path = f"quasideer_results_len{k}.csv"

    baseline_results_df.to_csv(main_results_csv_path, index=False)
    quasideer_results_df.to_csv(quasideer_results_csv_path, index=False)
