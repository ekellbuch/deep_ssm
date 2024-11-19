"""
Each array captures signals at a sampling frequency of 30kHz
The neural signal were binned into 20 millisecond intervals
so we ended up with a rate of 50 Hz (bins) per second.

Memory, it doesn't fit with batch_size 64 but it does fit with batch_size 32
Version which works, now try with lightning
"""
import hydra
import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor
from peft import LoraConfig, get_peft_model
import yaml
from omegaconf import OmegaConf, open_dict
from deep_ssm.data.data_loader import SpeechDataModule
from torchsummary import summary
import torch
import math
import torch.nn.functional as F
from torchaudio import transforms
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_metric
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.optim.lr_scheduler import LambdaLR
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
import wandb
from deep_ssm.modules.module_bci import BCIWhisperModule
from deep_ssm.utils.callbacks import all_callbacks

torch.autograd.set_detect_anomaly(True)

"""
def print_trainable_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameter: {name}, Shape: {param.shape}")
# Helper function to calculate trainable parameters
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
"""


class ModifiedWhisper(nn.Module):

    def __init__(self, in_channels, whisper_model):
        super(ModifiedWhisper, self).__init__()
        self.input_layer = nn.Conv1d(in_channels=in_channels, out_channels=80, kernel_size=3, stride=1, padding=1)
        self.whisper_model = whisper_model

    def forward(self, x, labels=None):
        # x: batch_size x length x num_features

        # Pass through modified input layer
        # Change shape to [batch, channels, time]

        # resample data and pad to 30 sec

        # x to shape batch_size, num_features, length
        # x = self.resample(x.permute(0, 2, 1).contiguous())

        # x to shape batch_size, length, out_channels
        x = self.input_layer(x.permute(0, 2, 1))

        # Forward through Whisper model
        outputs = self.whisper_model(input_features=x, labels=labels)
        return outputs


@hydra.main(config_path="../../configs/bci", config_name="baseline_whisper", version_base=None)
def main(args):
    # Load config file:
    #args = create_cfg()
    torch.set_float32_matmul_precision(args.matmul_precision)
    L.seed_everything(args.seed)

    # Load dataset
    datamodule = SpeechDataModule(args.data_cfg)
    datamodule.setup()

    # Load Whisper model and processor
    # extracts mel-filter bank features from raw speech using Short time Fourier Transform
    # feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(args.model_cfg.model_name)

    # loads whisper model
    model = WhisperForConditionalGeneration.from_pretrained(args.model_cfg.model_name)

    # Initialize modified model
    in_channels = args.model_cfg.configs.neural_dim
    modified_model = ModifiedWhisper(in_channels, model)

    # Freeze lower layers of the Whisper model encoder
    # or maybe here we should freeze the decoder?
    if args.model_cfg.configs.freeze_encoder:
        for param in modified_model.whisper_model.model.encoder.parameters():
            param.requires_grad = False

    if args.model_cfg.configs.freeze_decoder:
        for param in modified_model.whisper_model.model.decoder.parameters():
            param.requires_grad = False

    # Count and visualize the model parameters before LoRA
    print("Before applying LoRA:")
    #x_temp = torch.randn(256, 3000)
    #summary(modified_model, input_size=(1, 256, 3000), device='cpu')  # example input shape
    #print(f"Total trainable parameters: {count_trainable_params(modified_model)}")

    # Apply (Low-Rank Adaptation)
    if args.lora_cfg.apply_lora:

        # Get target modules for LoRA:
        target_modules = []
        for name, module in modified_model.named_modules():
            if isinstance(module, nn.Linear):
                target_modules.append(name)

        lora_config = LoraConfig(target_modules=target_modules,
                                **args.lora_cfg.configs)

        modified_model = get_peft_model(modified_model, lora_config)

        print("After applying LoRA:")
        #summary(modified_model, input_size=(1, 256, 3000))
        #print(f"Total trainable parameters after LoRA: {count_trainable_params(modified_model)}")

    # Define module
    modulito = BCIWhisperModule(args, modified_model, processor)

    # setup logger:
    if args.trainer_cfg.fast_dev_run:
      logger = None
    else:
      if args.trainer_cfg.logger == "wandb":
        if args.trainer_cfg.accelerator == "ddp":
          kwargs = {"group": "DDP"}
        else:
          kwargs = dict()

        logger = WandbLogger(name=args.experiment_name,
                             project=args.project_name, **kwargs)

        args_as_dict = OmegaConf.to_container(args)
        logger.log_hyperparams(args_as_dict)
      else:
        logger = None

    # set trainer:
    trainer_config = OmegaConf.to_container(args.trainer_cfg)
    trainer_config['logger'] = logger

    # set callbacks
    local_callbacks = []
    if args.callbacks:
      if args.callbacks.get("lr_monitor", None):
        local_callbacks.append(LearningRateMonitor(**args.callbacks.lr_monitor))
      if args.callbacks.get("grad_norm") and args.callbacks.grad_norm.get("type", None):
        local_callbacks.append(all_callbacks[args.callbacks.grad_norm.type])
      if args.callbacks.get("early_stopping", None):
        local_callbacks.append(EarlyStopping(**args.callbacks.early_stopping))
      if args.callbacks.get("masking_scheduler", None):
        local_callbacks.append(all_callbacks["masking_scheduler"](**args.callbacks.masking_scheduler))
        trainer_config["reload_dataloaders_every_n_epochs"] = 1
      if args.callbacks.get("checkpoint_cfg", None):
        local_callbacks.append(ModelCheckpoint(**args.callbacks.checkpoint_cfg))
      if args.callbacks.get("eigen_track", None):
        local_callbacks.append(all_callbacks["eigen_track"])

    trainer = L.Trainer(**trainer_config, callbacks=local_callbacks)

    # Train model
    if not args.eval_cfg.get("eval_only", 0):
      trainer.fit(model=modulito, datamodule=datamodule)
      ckpt_path = None
    else:
      ckpt_path = args.eval_cfg.get("ckpt_path", None)

    # Test model
    trainer.test(modulito, datamodule=datamodule, ckpt_path=ckpt_path)

    # End logging
    if args.trainer_cfg.logger == "wandb" and not (logger is None):
      wandb.run.summary["total_params"] = sum(p.numel() for p in model.parameters())
      wandb.finish()



if __name__ == "__main__":
    main()