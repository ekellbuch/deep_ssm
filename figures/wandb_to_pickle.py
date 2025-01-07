"""
wandb_to_pickle.py

helper file to save out wandb sweeps to pickle files
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
    data = []
    for run in tqdm(sweep.runs):
        run_data = run.history()
        run_data["run_id"] = run.id
        for key, value in run.config.items():
            run_data[key] = value
        data.append(run_data)

    df = pd.concat(data, ignore_index=True)
    print(df.shape)

    df.to_pickle(f"{args.pickle}_{args.sweep[-8:]}.pkl")
