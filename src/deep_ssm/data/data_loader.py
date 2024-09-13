import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
from torch.nn.utils.rnn import pad_sequence
from .data_transforms import AddWhiteNoise, AddOffset, SpeckleMasking, TemporalMasking, FeatureMasking
from torchvision.transforms import Compose


class SpeechDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        for day in range(self.n_days):
            for trial in range(len(data[day]["sentenceDat"])):
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                self.phone_seqs.append(data[day]["phonemes"][trial])
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                self.days.append(day)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        return (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )


def _padding(batch):
  X, y, X_lens, y_lens, days = zip(*batch)
  X_padded = pad_sequence(X, batch_first=True, padding_value=0)
  y_padded = pad_sequence(y, batch_first=True, padding_value=0)

  return (
    X_padded,
    y_padded,
    torch.stack(X_lens),
    torch.stack(y_lens),
    torch.stack(days),
  )


def getDatasetLoaders(args):
  with open(args.datasetPath, "rb") as handle:
    loadedData = pickle.load(handle)


  transforms = []
  if args.feature_mask_p > 0:
    transforms.append(FeatureMasking(args.feature_mask_p, args.mask_value))
  if args.temporal_mask_n > 0:
    transforms.append(TemporalMasking(args.temporal_mask_n,args.mask_value, args.temporal_mask_len))
  if args.speckled_mask_p > 0:
    transforms.append(SpeckleMasking(args.speckled_mask_p, args.mask_value, args.renormalize_masking))
  if args.whiteNoiseSD > 0:
    transforms.append(AddWhiteNoise(args.whiteNoiseSD))
  if args.constantOffsetSD > 0:
    transforms.append(AddOffset(args.constantOffsetSD))

  if len(transforms) > 0:
    transform_fn = Compose(transforms)
  else:
    transform_fn = None

  train_ds = SpeechDataset(loadedData["train"], transform=transform_fn)
  test_ds = SpeechDataset(loadedData["test"])

  train_loader = DataLoader(
    train_ds,
    batch_size=args.batchSize,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    collate_fn=_padding,
  )
  test_loader = DataLoader(
    test_ds,
    batch_size=args.batchSize,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
    collate_fn=_padding,
  )

  return train_loader, test_loader, loadedData


