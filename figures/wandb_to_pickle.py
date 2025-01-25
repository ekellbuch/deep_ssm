"""
wandb_to_pickle.py

helper file to save out wandb sweeps to pickle files

Usage:

name of sweep: user/project/sweep_id
name of pickle file: a helpful name for the pickle file
"""

from tqdm import tqdm
import pandas as pd
import argparse
import wandb
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, required=True, help="Name of the sweep")
    parser.add_argument("--pickle", type=str, required=True, help="Name of pickle file")
    args = parser.parse_args()

    api = wandb.Api()
    sweep = api.sweep(args.sweep)
    dataframes = []
    for run in tqdm(sweep.runs):
        history = []
        for row in run.scan_history():  # important to get all the run information out
            history.append(row)
        run_data = pd.DataFrame(history)
        for key, value in run.config.items():
            run_data[key] = value  # Add configuration as columns
        dataframes.append(run_data)

    df = pd.concat(dataframes, ignore_index=True)
    print(df.shape)

    df.to_pickle(f"{args.pickle}_{args.sweep[-8:]}.pkl")
