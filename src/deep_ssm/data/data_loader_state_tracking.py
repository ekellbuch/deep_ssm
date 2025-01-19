import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import os
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Compose
import math
import torch.nn.functional as F
from torch import Tensor
import lightning as L
from ordered_set import OrderedSet
from functools import partial
from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from typing import Optional, Union

# from enum import StrEnum
from pathlib import Path
import polars as pl


class StateTrackingDataset(Dataset):
    def __init__(self, input_sequences, output_sequences):
        """
        Initialize the dataset with lists of input and output sequences.

        Args:
            input_sequences (list): List of strings containing '0's and '1's
            output_sequences (list): List of strings containing '0's and '1's
        """
        assert len(input_sequences) == len(
            output_sequences
        ), "Input and output sequences must have same length"

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
    # input_masks = torch.zeros_like(padded_inputs, dtype=torch.bool)
    # output_masks = torch.zeros_like(padded_outputs, dtype=torch.bool)
    masks = torch.zeros_like(padded_inputs, dtype=torch.bool)

    for i, (inp_seq, out_seq) in enumerate(zip(input_sequences, output_sequences)):
        # input_masks[i, :len(inp_seq)] = 1
        # output_masks[i, :len(out_seq)] = 1
        masks[i, : len(inp_seq)] = 1

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
        with open(os.path.join(self.args.datasetPath, "train_src.txt"), "r") as f:
            train_src = [line.strip() for line in f.readlines()]
        with open(os.path.join(self.args.datasetPath, "train_tgt.txt"), "r") as f:
            train_tgt = [line.strip() for line in f.readlines()]
        with open(os.path.join(self.args.datasetPath, "val_src_bin0.txt"), "r") as f:
            val_src = [line.strip() for line in f.readlines()]
        with open(os.path.join(self.args.datasetPath, "val_tgt_bin0.txt"), "r") as f:
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
            collate_fn=collate_sequences,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.args.batchSize,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collate_sequences,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.args.batchSize,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collate_sequences,
        )

class SpecialToken(str):
    """Represents a special token with a value."""
    def __new__(cls, value: str):
        return str.__new__(cls, value)

    def __init__(self, value: str):
        self.value = value


class SpecialTokens:
    """Special tokens for tokenizer."""
    PAD = SpecialToken("[PAD]")
    BOS = SpecialToken("[BOS]")
    UNK = SpecialToken("[UNK]")
    EOS = SpecialToken("[EOS]")
    SEP = SpecialToken("[SEP]")
    CLS = SpecialToken("[CLS]")
    MASK = SpecialToken("[MASK]")

    @classmethod
    def values(cls):
        """Return a list of the string values of each special token."""
        return [str(attr) for attr in cls.__dict__.values() if isinstance(attr, SpecialToken)]

    @classmethod
    def index(cls, token: str):
        """Return the index of the token in the list of special tokens."""
        values = cls.values()
        if token not in values:
            raise ValueError(f"Token '{token}' not found in special tokens.")
        return values.index(token)


def pad_collate(
    samples: list[dict[str, Tensor]], pad_token_id: int
) -> dict[str, Tensor]:
    """Collate function for DataLoader.

    Performs channel-wise padding of the inputs and targets.
    """
    max_len = max([s["input_ids"].shape[0] for s in samples])
    masks = torch.zeros((len(samples), max_len), dtype=torch.bool)
    for i, s in enumerate(samples):
        masks[i, :s["input_ids"].shape[0]] = 1
        if max_len > s["input_ids"].shape[0]:
            s["input_ids"] = F.pad(s["input_ids"], (0, max_len - s["input_ids"].shape[0]), value=pad_token_id)
            s["labels"] = F.pad(s["labels"], (0, max_len - s["labels"].shape[0]), value=pad_token_id)

    input_ids = torch.stack([s["input_ids"] for s in samples])
    labels = torch.stack([s["labels"] for s in samples])

    return input_ids, labels, masks


def tokenize(
    example: dict[str, Tensor],
    tokenizer: PreTrainedTokenizerFast,
    supervised: bool,
) -> dict[str, Tensor]:
    """Tokenize inputs."""
    tokenized = tokenizer(
        example["input"],
        return_tensors="pt",
        padding=True,
    )
    tokenized.pop("attention_mask", None)

    # If output is not supervised (e.g., for MLPs) then we only keep the final target
    # value since its sequence classification, not token classification.
    tokenized["labels"] = tokenizer(
        example["target"],
        return_tensors="pt",
        padding=True,
    )["input_ids"]
    if not supervised:
        tokenized["labels"] = tokenized["labels"][:, -1]

    return tokenized


def get_dataset(
    group: str,
    k: int,
    strict_len: bool,
    train_size: float,
    data_dir: str,
    supervised: bool = True,
    max_samples: Optional[int] = None,
) -> dict:
    """Construct dataset."""
    assert train_size > 0 and train_size <= 1, "`train_size` must be in (0,1]"

    if strict_len:
        assert k > 1, "`k` must be at least 2"
        data_paths = [os.path.join(data_dir, f"{group}={i}.csv") for i in [2, k]]
        data_paths = list(OrderedSet(data_paths))
        if not os.path.exists(data_paths[0]):
            raise FileNotFoundError(f"You must have data for {group}={2}.")
    else:
        data_paths = [
            os.path.join(data_dir, f"{group}={i}.csv") for i in range(2, k + 1)
        ]
        if not os.path.exists(data_paths[0]):
            raise FileNotFoundError(f"You must have data for {group}=2.")
        data_paths = [p for p in data_paths if os.path.exists(p)]
        data_paths = list(OrderedSet(data_paths))

    # All unique tokens can be found by looking at the k=2 inputs. We create a
    # a dictionary mapping each token to its index in the vocabulary and use this
    # to construct the tokenizer.
    unique_tokens = (
        pl.read_csv(data_paths[0])
        .select(pl.col("input").map_batches(lambda x: x.str.split(" ")))
        .explode("input")
        .unique()["input"]
        .to_list()
    )
    unique_tokens = {t: int(t) for t in unique_tokens}

    tokenizer_base = Tokenizer(WordLevel())
    tokenizer_base.pre_tokenizer = WhitespaceSplit()
    tokenizer_base.add_tokens(sorted(list(unique_tokens.keys()), key=lambda x: int(x)))
    tokenizer_base.add_special_tokens(SpecialTokens.values())
    tokenizer_base.post_processor = TemplateProcessing(
        single=f"{SpecialTokens.BOS} $A",
        special_tokens=[
            (SpecialTokens.BOS, tokenizer_base.token_to_id(SpecialTokens.BOS))
        ],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_base,
        bos_token=SpecialTokens.BOS.value,
        unk_token=SpecialTokens.UNK.value,
        eos_token=SpecialTokens.EOS.value,
        sep_token=SpecialTokens.SEP.value,
        cls_token=SpecialTokens.CLS.value,
        mask_token=SpecialTokens.MASK.value,
        pad_token=SpecialTokens.PAD.value,
    )
    tokenizer.padding_side = "right"
    tokenize_map = partial(tokenize, tokenizer=tokenizer, supervised=supervised)

    # Construct dataset
    if len(data_paths) == 1:
        dataset = (
            load_dataset("csv", data_files=str(data_paths[0]), split="all")
            .remove_columns(["seed"])
            .map(tokenize_map, batched=True)
            .remove_columns(["input", "target", "token_type_ids"])
        )
        if max_samples is not None:
            num_samples = min(len(dataset), max_samples)
            dataset = dataset.select(range(num_samples))
        if train_size < 1:
            dataset = dataset.train_test_split(train_size=train_size)
    else:
        train_data = [
            load_dataset("csv", data_files=str(d_path), split="all")
            .remove_columns(["seed"])
            .map(tokenize_map, batched=True)
            .remove_columns(["input", "target", "token_type_ids"])
            for d_path in data_paths[:-1]
        ]
        k_data = (
            load_dataset("csv", data_files=str(data_paths[-1]), split="all")
            .remove_columns(["seed"])
            .map(tokenize_map, batched=True)
            .remove_columns(["input", "target", "token_type_ids"])
        )

        if max_samples is not None:
            k_data = k_data.select(range(min(len(k_data), max_samples)))
            train_data = [t.select(range(min(len(t), max_samples))) for t in train_data]

        train_data = concatenate_datasets(train_data)

        if train_size < 1:
            dataset = k_data.train_test_split(train_size=train_size)
            dataset["train"] = concatenate_datasets([dataset["train"], train_data])
        else:
            dataset = concatenate_datasets([train_data, k_data])

    return {
        "dataset": dataset.with_format("torch"),
        "tokenizer": tokenizer,
        "n_vocab": tokenizer_base.get_vocab_size(with_added_tokens=True),
    }


class GroupMultiplicationDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        datadict = get_dataset(
            group=self.args.group,
            k=self.args.k,
            strict_len=True,
            train_size=0.8,
            data_dir=self.args.datasetPath,
            supervised=True,
            max_samples=self.args.max_samples,
        )
        dataset = datadict["dataset"]
        self.n_vocab = datadict["n_vocab"]
        tokenizer = datadict["tokenizer"]
        self.collate_fn = partial(pad_collate, pad_token_id=tokenizer.pad_token_id)

        # Create train, validation, and test datasets
        self.train_ds = dataset["train"]
        self.val_ds = dataset["test"]
        self.test_ds = dataset["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.args.batchSize,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.args.batchSize,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.args.batchSize,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
