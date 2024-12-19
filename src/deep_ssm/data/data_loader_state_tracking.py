import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import os
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Compose
import math
import torch.nn.functional as F
import lightning as L


class StateTrackingDataset(Dataset):
    def __init__(self, input_sequences, output_sequences):
        """
        Initialize the dataset with lists of input and output sequences.

        Args:
            input_sequences (list): List of strings containing '0's and '1's
            output_sequences (list): List of strings containing '0's and '1's
        """
        assert len(input_sequences) == len(output_sequences), "Input and output sequences must have same length"

        # Convert strings to lists of integers, then to tensors
        self.input_sequences = [
            torch.tensor([int(c) for c in seq], dtype=torch.long)
            for seq in input_sequences
        ]
        self.output_sequences = [
            torch.tensor([int(c) for c in seq], dtype=torch.long)
            for seq in output_sequences
        ]

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.output_sequences[idx]

def collate_sequences(batch):
    """
    Custom collate function for DataLoader that handles padding and mask creation.

    Args:
        batch: List of tuples (input_seq, output_seq)

    Returns:
        tuple: (padded_inputs, input_masks, padded_outputs, output_masks)
    """
    # Separate inputs and outputs
    input_sequences, output_sequences = zip(*batch)

    # Pad sequences
    padded_inputs = pad_sequence(input_sequences, batch_first=True, padding_value=0)
    padded_outputs = pad_sequence(output_sequences, batch_first=True, padding_value=0)

    # Create masks (1 for actual sequence, 0 for padding)
    #input_masks = torch.zeros_like(padded_inputs, dtype=torch.bool)
    #output_masks = torch.zeros_like(padded_outputs, dtype=torch.bool)
    masks = torch.zeros_like(padded_inputs, dtype=torch.bool)

    for i, (inp_seq, out_seq) in enumerate(zip(input_sequences, output_sequences)):
        #input_masks[i, :len(inp_seq)] = 1
        #output_masks[i, :len(out_seq)] = 1
        masks[i, :len(inp_seq)] = 1

    return padded_inputs, padded_outputs, masks

class ParityDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        # Load the dataset from the txt files
        with open(os.path.join(self.args.datasetPath, 'train_src.txt'), 'r') as f:
            train_src = [line.strip() for line in f.readlines()]
        with open(os.path.join(self.args.datasetPath, 'train_tgt.txt'), 'r') as f:
            train_tgt = [line.strip() for line in f.readlines()]
        with open(os.path.join(self.args.datasetPath, 'val_src_bin0.txt'), 'r') as f:
            val_src = [line.strip() for line in f.readlines()]
        with open(os.path.join(self.args.datasetPath, 'val_tgt_bin0.txt'), 'r') as f:
            val_tgt = [line.strip() for line in f.readlines()]

        # Create train, validation, and test datasets
        self.train_ds = StateTrackingDataset(train_src, train_tgt)
        self.val_ds = StateTrackingDataset(val_src, val_tgt)
        self.test_ds = StateTrackingDataset(val_src, val_tgt)


    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.args.batchSize,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collate_sequences
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.args.batchSize,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collate_sequences
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.args.batchSize,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collate_sequences
        )

class GroupMultiplicationDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        # Load the dataset from the txt files
        with open(os.path.join(self.args.datasetPath, 'train_src.txt'), 'r') as f:
            train_src = [line.strip() for line in f.readlines()]
        with open(os.path.join(self.args.datasetPath, 'train_tgt.txt'), 'r') as f:
            train_tgt = [line.strip() for line in f.readlines()]
        with open(os.path.join(self.args.datasetPath, 'val_src_bin0.txt'), 'r') as f:
            val_src = [line.strip() for line in f.readlines()]
        with open(os.path.join(self.args.datasetPath, 'val_tgt_bin0.txt'), 'r') as f:
            val_tgt = [line.strip() for line in f.readlines()]

        # Create train, validation, and test datasets
        self.train_ds = StateTrackingDataset(train_src, train_tgt)
        self.val_ds = StateTrackingDataset(val_src, val_tgt)
        self.test_ds = StateTrackingDataset(val_src, val_tgt)


    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.args.batchSize,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collate_sequences
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.args.batchSize,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collate_sequences
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.args.batchSize,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collate_sequences
        )
