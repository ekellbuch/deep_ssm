"""

"""
from deep_ssm.data.data_loader import SpeechDataset, getDatasetLoaders
import torch
from tqdm import tqdm
from deep_ssm.models import all_models
import yaml
from omegaconf import OmegaConf, open_dict
import numpy as np
import pickle
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment in https://wandb.ai/ekellbuch/speech_bci/runs/99p7m247/overview
TOY_CONFIG = "deepseq/deep_ssm/configs/bci/baseline_mamba.yaml"
checkpoint = "deepseq/deep_ssm/experiments/speech_bci/99p7m247/checkpoints/epoch=72-step=10000.ckpt"


# read baseline_mamba.yaml
def create_cfg() -> dict:
  """Load all toy data config file without hydra."""
  cfg = yaml.load(open(str(TOY_CONFIG)), Loader=yaml.FullLoader)
  return OmegaConf.create(cfg)


def main():
  args = create_cfg()

  # read Data:
  train_loader, test_loader, loadedData = getDatasetLoaders(args.data_cfg)

  # get model:
  model = all_models[args.model_cfg.type](**args.model_cfg.configs, nDays=len(loadedData["train"]))

  # load checkpoint:
  weights = torch.load(checkpoint)['state_dict']
  weights = {k.replace('model.', ''): v for k, v in weights.items()}

  model.load_state_dict(weights)
  model = model.to(device)

  # Eval mode
  model.eval()

  # Get competition Data
  partition = "competition"
  mamba_outputs = {
    "logits": [],
    "logitLengths": [],
    "trueSeqs": [],
    "transcriptions": [],
  }
  for i, testDayIdx in tqdm(enumerate([4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20])):
    test_ds = SpeechDataset([loadedData[partition][i]])
    test_loader = torch.utils.data.DataLoader(
      test_ds, batch_size=1, shuffle=False, num_workers=0
    )
    for j, (X, y, X_len, y_len, _) in enumerate(test_loader):
      X, y, X_len, y_len, dayIdx = (
        X.to(device),
        y.to(device),
        X_len.to(device),
        y_len.to(device),
        torch.tensor([testDayIdx], dtype=torch.int64).to(device),
      )

      # get prediction
      pred = model.forward(X, dayIdx)
      adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

      for iterIdx in range(pred.shape[0]):
        trueSeq = np.array(y[iterIdx][0: y_len[iterIdx]].cpu().detach())

        mamba_outputs["logits"].append(pred[iterIdx].cpu().detach().numpy())
        mamba_outputs["logitLengths"].append(
          adjustedLens[iterIdx].cpu().detach().item()
        )
        mamba_outputs["trueSeqs"].append(trueSeq)

      transcript = loadedData[partition][i]["transcriptions"][j].strip()
      transcript = re.sub(r"[^a-zA-Z\- \']", "", transcript)
      transcript = transcript.replace("--", "").lower()
      mamba_outputs["transcriptions"].append(transcript)

  pickle.dump(mamba_outputs, open(f'model_prediction_{partition}.pkl', 'wb'))


if __name__ == "__main__":
  main()