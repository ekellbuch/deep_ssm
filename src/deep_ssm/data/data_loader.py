import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Compose
import math
import torch.nn.functional as F
from deep_ssm.data.data_transforms import AddWhiteNoise, AddOffset, SpeckleMasking, TemporalMasking, FeatureMasking, DataResample
import lightning as L
from transformers import WhisperTokenizer
import re


class SpeechDataset(Dataset):
    def __init__(self, data, transform=None, processor=None, return_text=False):
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.return_text = return_text
        self.processor = processor
        self.tokenizer = self.get_tokenizer()

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.transcriptions = []
        self.days = []

        for day in range(self.n_days):
            for trial in range(len(data[day]["sentenceDat"])):
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                self.phone_seqs.append(data[day]["phonemes"][trial])
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                self.days.append(day)
                if self.return_text:
                  self.transcriptions.append(data[day]["transcriptions"][trial])

    def __len__(self):
        return self.n_trials

    def get_tokenizer(self):
      if self.processor == "whisper":
        return WhisperTokenizer.from_pretrained("openai/whisper-small.en", task="transcribe")
      else:
        return None

    def get_transcription(self, idx):
      if self.tokenizer is not None:
          transcript = self.transcriptions[idx]
          # process transcript: 
          transcript = transcript.strip()
          transcript = re.sub(r"[^a-zA-Z\- \']", "", transcript)
          transcript = transcript.replace("--", "").lower()
          transcription = self.tokenizer(text=transcript, return_tensors="pt",
                                       padding=True).input_ids.squeeze(0)
      else:
        transcription = self.transcriptions[idx]
      return transcription

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        if self.return_text:
            return (
                neural_feats,
                torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
                torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
                torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
                torch.tensor(self.days[idx], dtype=torch.int64),
                self.get_transcription(idx),
            )
        else:
          return (
              neural_feats,
              torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
              torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
              torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
              torch.tensor(self.days[idx], dtype=torch.int64),
          )


def _padding(batch, padding_type=None, padding_len=0):
  if len(batch[0]) == 6:
    X, y, X_lens, y_lens, days, transcriptions = zip(*batch)

  else:
    X, y, X_lens, y_lens, days = zip(*batch)

  max_len = max(seq.size(0) for seq in X)

  # Pad the sequences:
  # use default padding, i.e. all sequences in batch are the same length
  X_padded = pad_sequence(X, batch_first=True, padding_value=0)

  if padding_type == 'multiple':
    desired_len = math.ceil(max_len / multiple) * padding_len
    X_padded = F.pad(X_padded, (0, 0, 0,  desired_len - X_padded.size(1)), value=0)
  elif padding_type == 'fixed_len':
    X_padded = F.pad(X_padded, (0, 0, 0,  padding_len - X_padded.size(1)), value=0)

  y_padded = pad_sequence(y, batch_first=True, padding_value=0)

  if len(batch[0]) == 5:
    return (
      X_padded,
      y_padded,
      torch.stack(X_lens),
      torch.stack(y_lens),
      torch.stack(days),
    )
  else:
    transcriptions_padded = pad_sequence(transcriptions, batch_first=True, padding_value=-100)
    return (
      X_padded,
      y_padded,
      torch.stack(X_lens),
      torch.stack(y_lens),
      torch.stack(days),
      transcriptions_padded
    )


def get_data_augmentations(args):
  transforms = []
  if args.get("feature_mask_p", 0) > 0:
    transforms.append(FeatureMasking(args.feature_mask_p, args.mask_value))
  if args.get("temporal_mask_n", 0) > 0:
    transforms.append(TemporalMasking(args.temporal_mask_n,args.mask_value, args.temporal_mask_len))
  if args.get("speckled_mask_p", 0) > 0:
    transforms.append(SpeckleMasking(args.speckled_mask_p, args.mask_value, args.renormalize_masking))
  if args.get("whiteNoiseSD", 0) > 0:
    transforms.append(AddWhiteNoise(args.whiteNoiseSD))
  if args.get("constantOffsetSD", 0) > 0:
    transforms.append(AddOffset(args.constantOffsetSD))
  if args.get("resample", False):
    transforms.append(DataResample(args.orig_freq, args.new_freq))
  if len(transforms) > 0:
    transform_fn = Compose(transforms)
  else:
    transform_fn = None
  return transform_fn


def get_test_augmentations(args):
  # only a subset of augmentations are passed at test time 
  transforms = []
  if args.get("resample", False):
    transforms.append(DataResample(args.orig_freq, args.new_freq))
  if len(transforms) > 0:
    transform_fn = Compose(transforms)
  else:
    transform_fn = None
  return transform_fn



class SpeechDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.loadedData = None
        # TODO: clean this up
        self.nDays = 24
        self.return_text = args.get("return_text", False)
        self.processor = args.get("processor", None)

    def setup(self, stage=None):
        # Load the dataset from the pickle file
        with open(self.args.datasetPath, "rb") as handle:
            loadedData = pickle.load(handle)

        self.nDays = len(loadedData["train"])
        # Set up the fdata transforms
        transform_fn = get_data_augmentations(self.args)

        # Create train, validation, and test datasets
        train_ds = SpeechDataset(loadedData["train"],
                                 transform=transform_fn,
                                 processor=self.processor,
                                 return_text=self.return_text)

        transform_fn_test = get_test_augmentations(self.args)
        
        test_ds = SpeechDataset(loadedData["test"],
                                processor=self.processor,
                                return_text=self.return_text)

        if self.args.get("padding_type", None):
          padding_type = self.args.get("padding_type", None)
          padding_len = self.args.get("padding_len", 0)
          self.padding_fn = lambda x: _padding(x, padding_type=padding_type, padding_len=padding_len)
        else:
          self.padding_fn = _padding

        # Split train and validation datasets if needed
        if self.args.get("train_split", 1) < 1:
          train_len = int(len(train_ds) * self.args.train_split)
          val_len = len(train_ds) - train_len
          train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_len, 1 - val_len])
          val_ds.dataset.transform = None
          self.train_ds = train_ds
          self.val_ds = val_ds
        else:
            self.train_ds, self.val_ds = train_ds, test_ds

        self.test_ds = test_ds

    def update_transforms(self):
        transform_fn = get_data_augmentations(self.args)
        # Update transform function
        if isinstance(self.train_ds, torch.utils.data.Subset):
            self.train_ds.dataset.transform = transform_fn
        else:
            self.train_ds.transform = transform_fn
        print(f"Set new loader with params {self.args}")


    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.args.batchSize,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=self.padding_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.args.batchSize,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=self.padding_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.args.batchSize,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=self.padding_fn,
        )