"""
mnist_unrolling_results_base.csv results:

Ran seed= 1, clip=True, num_train_iter=1, num_eval_iter=1 test_acc=0.621999979019165, test_loss=1.0766233140945434
Ran seed= 1, clip=True, num_train_iter=3, num_eval_iter=3 test_acc=0.9721999764442444, test_loss=0.09100578255653381
Ran seed= 1, clip=True, num_train_iter=5, num_eval_iter=5 test_acc=0.9803999662399292, test_loss=0.0624376249441877
Ran seed= 1, clip=True, num_train_iter=7, num_eval_iter=7 test_acc=0.9792999625205994, test_loss=0.0641708436600864

"""
import lightning as L
import torch.nn as nn
import numpy as np

import wandb
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from deep_ssm.data.basic import load_sequential_pixels_all
from deep_ssm.models.time_series_classifiers import pRNNClassifier
import torch
import os

try:
    import torch_xla.core.xla_model as xm
    DEVICE = xm.xla_device()  # TPU
except:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")  # macOS Metal backend
        compile_backend="aot_eager"
        compile_backend="eager"
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")  # GPU
        compile_backend = None
    else:
        DEVICE = torch.device("cpu")  # CPU fallback
        compile_backend = None


try:
    import torch_xla.core.xla_model as xm

    DEVICE = xm.xla_device()  # TPU
except:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")  # macOS Metal backend
        compile_backend = "aot_eager"
        compile_backend = "eager"
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")  # GPU
        compile_backend = None
    else:
        DEVICE = torch.device("cpu")  # CPU fallback
        compile_backend = None


def eval_loop(model, test_loader, device, criterion):
    test_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.permute(0, 2, 1)  # (B, T, D)
            logits = model(x)
            predictions = torch.argmax(logits, dim=1)
            loss = criterion(logits, y)
            test_loss += loss.item() * x.size(0)
            correct += torch.sum(predictions == y)
            total += len(y)

    test_loss /= total
    test_acc = correct / total
    return test_acc, test_loss

def main():
    # https://wandb.ai/xavier_gonzalez/nonlinear_ssm/sweeps/n72ykb80?nw=nwuserxavier_gonzalez

    # Initialize the API
    api = wandb.Api()
    # Define your sweep path
    # run: https://wandb.ai/xavier_gonzalez/nonlinear_ssm/sweeps/n72ykb80/overview
    # parameters:
    # clip = True, False
    # num_train_iters = range(1, 10)
    # seeds = [1,2,3]

    # total number of evaluations is 60
    CKPT_BASE = '/scratch/groups/swl1/xg_deep_ssm_checkpoints_01_22/checkpoints'

    sweep_path = "xavier_gonzalez/nonlinear_ssm/n72ykb80"

    # Fetch the sweep
    sweep = api.sweep(sweep_path)

    # Retrieve all runs associated with the sweep
    runs = sweep.runs

    # Extract relevant data
    summary_list = []
    config_list = []
    name_list = []
    dirs_list = []

    for run in runs:
        # Summary metrics
        summary_list.append(run.summary._json_dict)
        # Hyperparameters
        config_list.append({k: v for k, v in run.config.items() if not k.startswith('_')})
        # Run name
        name_list.append(run.name)
        dirs_list.append(run.dir.rsplit('/')[-1])

    # Create a DataFrame
    runs_df = pd.DataFrame({
        "name": name_list,
        "config": config_list,
        "summary": summary_list,
        "dirs": dirs_list
    })

    all_seeds = [run_['seed'] for run_ in runs_df.config]
    all_train_iters = [run_['model_cfg']['num_iters'] for run_ in runs_df.config]
    all_clips = [run_['model_cfg']['clip'] for run_ in runs_df.config]
    all_ks = [run_['model_cfg']['k'] for run_ in runs_df.config]
    all_ckpt_dirs = [run_ for run_ in runs_df.dirs]

    runs_df2 = pd.DataFrame({
        "seed": all_seeds,
        "train_iter": all_train_iters,
        "clip": all_clips,
        "dirs": all_ckpt_dirs,
    })

    all_clips = np.unique(all_clips)
    all_seeds = np.unique(all_seeds)
    all_train_iters = np.unique(all_train_iters)[::2]
    all_eval_iters = [10] #all_train_iters.tolist() + [11, 15, 20]

    # only look at runs_df2
    # runs_df2 = runs_df2[runs_df2['clip'] == True]
    # Load test time code and then run
    all_outputs = []
    for clip in [True]:
        for seed in [1]:  # all_seeds
            # myrun = runs_df2[runs_df2['seed'] ==seed]
            L.seed_everything(seed)
            # it wasn't artifact:
            wandb.init(mode='disabled')
            args = OmegaConf.load("../configs/seq_mnist/seq_mnist.yaml")
            args.seed = seed
            print(args)
            # Load datasets
            cfg = args
            num_epochs = cfg.trainer_cfg.num_epochs
            num_iters = cfg.model_cfg.num_iters
            batch_size = cfg.data_cfg.batch_size
            learning_rate = cfg.optimizer_cfg.lr
            num_layers = cfg.model_cfg.num_layers
            use_parallel = cfg.model_cfg.parallel
            offline = not cfg.logger_cfg.use_wandb

            train_loader, val_loader, test_loader = load_sequential_pixels_all(
                batch_size=batch_size, val_split=0.1, seed=cfg.seed, explore=cfg.data_cfg.explore,
                dataset_name=cfg.data_cfg.dataset_name)

            if cfg.data_cfg.dataset_name == 'mnist':
                d_input = 1
            elif cfg.data_cfg.dataset_name == 'cifar':
                d_input = 3
            else:
                raise ValueError(f"Unknown dataset: {cfg.data_cfg.dataset_name}")

            hidden_size = cfg.model_cfg.hidden_size
            while_loop = False

            # Now let's load a checkpoint
            # num_train_iters = range(1, 10)
            # num_eval_iters =  [1, 15]
            for num_train_iter in [3]: #all_train_iters:
                # Load checkpoint
                mask = (runs_df2['seed']) == seed & (runs_df2['clip'] == clip) & (runs_df2['train_iter'] == num_train_iter)
                checkpoint = runs_df2[mask]['dirs'].values[0]
                #ckpt_path = os.path.join(CKPT_BASE, checkpoint, 'best_model.pth')
                #breakpoint()
                ckpt_path = os.path.join('best_model.pth')
                weights = torch.load(ckpt_path,weights_only=True,map_location=torch.device('cpu'))#['state_dict']
                #weights = {k.replace('model.', ''): v for k, v in weights.items()}
                #print(ckpt_path)
                #for num_eval_iter in all_eval_iters:
                for num_eval_iter in [4]:
                    # Load model
                    model = pRNNClassifier(
                        input_size=d_input,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        num_iters=num_eval_iter,
                        parallel=use_parallel,
                        learn_diags=cfg.model_cfg.learn_diags,
                        k=0.5,#cfg.model_cfg.k,
                        clip=clip,
                        method=cfg.model_cfg.method,
                        last_only=cfg.model_cfg.last_only,
                        learn_initial_states=cfg.model_cfg.learn_initial_states,
                        while_loop=while_loop,
                        quasi=cfg.model_cfg.quasi,
                        family=cfg.model_cfg.family,
                        use_mean=cfg.model_cfg.use_mean,
                        conv=cfg.model_cfg.conv,
                        num_classes=10,
                        clip_post=False,
                    )

                    criterion = nn.CrossEntropyLoss()
                    model.load_state_dict(weights)

                    model = model.to(DEVICE)
                    model.eval()
                    torch.compile(model, backend=compile_backend)
                    test_loader = [next(iter(test_loader))]
                    test_acc, test_loss = eval_loop(model, test_loader, DEVICE, criterion)
                    print(f"Ran seed= {seed}, clip={clip}, num_train_iter={num_train_iter}, num_eval_iter={num_eval_iter} test_acc={test_acc.item()}, test_loss={test_loss}")
                    #breakpoint()
                    c_output = {"test_acc": test_acc.item(),
                                "test_loss": test_loss,
                                "clip": clip,
                                "num_train_iter": num_train_iter,
                                "num_eval_iter": num_eval_iter,
                                "seed" : seed,
                                }
                    all_outputs.append(c_output)

    # Finally save all results:
    baseline_results_df = pd.DataFrame(all_outputs)

    #print(baseline_results_df)
    #main_results_csv_path = f"mnist_unrolling_results2.csv"

    #baseline_results_df.to_csv(main_results_csv_path, index=False)


if __name__ == "__main__":
    main()