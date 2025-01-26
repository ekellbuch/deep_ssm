"""

"""
import wandb
import os
from run_group_multiplication import train
from omegaconf import OmegaConf

def get_resume_sweep_configs(sweep_path):
    # Initialize the API
    api = wandb.Api()
    # Fetch the sweep
    sweep = api.sweep(sweep_path)
    # Retrieve all runs associated with the sweep
    runs = sweep.runs
    all_new_run_configs = []
    for run in runs:
        new_run_configs = get_new_run_configs(run)
        if new_run_configs is not None:
            all_new_run_configs.append(new_run_configs)

    print("Number of runs to resume: ", len(all_new_run_configs))
    return all_new_run_configs

def get_run_checkpoint_path(run):
    try:
        output_dir = run.summary['output_dir']
        ckpt_dir = os.path.join(output_dir, f"{run.project}/{run.id}/checkpoints")
        # read ckpt_dir
        ckpt_path = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0])
        assert os.path.exists(ckpt_path)
    except:
        raise NotImplementedError("include where you stored checkpoints")
    return ckpt_path

def get_new_run_configs(run):
    # Read run checkpoint path and run configs
    ckpt_path = get_run_checkpoint_path(run)
    #print(run.state)
    #if not(run.state is "finished"):
    #    print("Experiment has finished")
    #    return None
    # Load experiment configs:
    configs = run.config
    configs['trainer_fit_cfg'] = {}
    configs['trainer_fit_cfg']['ckpt_path'] = ckpt_path
    configs = OmegaConf.create(configs)
    return configs

def get_resume_run_configs(run_path):
    # Initialize the API
    api = wandb.Api()
    # Fetch the run:
    run = api.run(run_path)
    # Read config given run path
    configs =  get_new_run_configs(run)
    return [configs]

if __name__ == "__main__":
    #sweep_path = "xavier_gonzalez/nonlinear_ssm/n72ykb80"
    #configs = get_resume_sweep_configs(sweep_path,)
    run_path = "ekellbuch/group_multiplication/zrabz9ta"
    configs = get_resume_run_configs(run_path)

    # Create wandb run given each config:
    # somehow we want to do this in parallel and submit a bunch of jobs:
    for config in configs:
        run = wandb.init(config=config)
        train(config)

    # need to find cli