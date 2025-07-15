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
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.optim.lr_scheduler import LambdaLR
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
import wandb
from deep_ssm.modules.module_bci import BCIWhisperModule
from deep_ssm.utils.callbacks import all_callbacks
from deep_ssm.data.data_transforms import GaussianSmoothing

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
    def __init__(
        self,
        in_channels,
        whisper_model,
        gaussian_smooth_width=2.0,
        patch_size=5,
        apply_time_masking=True,
        max_len=1500,
        dropout=0.2,
        kernel_size=21,
    ):
        super(ModifiedWhisper, self).__init__()

        self.whisper_model = whisper_model
        self.apply_time_masking = apply_time_masking
        self.max_len = max_len

        # BiT-style temporal smoothing
        self.temporal_smooth = GaussianSmoothing(
            channels=in_channels,
            kernel_size=kernel_size,
            sigma=gaussian_smooth_width,
            dim=1,
        )

        # Patch-style projection: [B, C, T] -> [B, 80, T_patched]
        self.feature_projection = nn.Sequential(
            nn.Conv1d(in_channels, 512, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(256, 80, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.post_projection_norm = nn.LayerNorm(80)

    def forward(self, x, labels=None):
      # x: [B, T, F]
      B, T, D = x.shape

      # Smooth across time dimension (permute to [B, F, T])
      x = x.permute(0, 2, 1)                        # [B, F, T]
      x = self.temporal_smooth(x)                   # [B, F, T]
      x = self.feature_projection(x)                # [B, 80, T']
      x = x.permute(0, 2, 1)                # [B, T, 80] 80 is last dim now
      x = self.post_projection_norm(x)      # valid input to LayerNorm
      x = x.permute(0, 2, 1)                # [B, 80, T] back to Whisper format

      # Whisper expects [B, 80, 3000]
      expected_len = 3000
      T_current = x.shape[-1]

      if T_current < expected_len:
          pad_len = expected_len - T_current
          x = F.pad(x, (0, pad_len), mode="constant")  # pad time axis (last dim)
      elif T_current > expected_len:
          x = x[:, :, :expected_len]  # truncate time axis

      # Apply simple time masking (SpecAugment-style)
      if self.training and self.apply_time_masking:
          x = self._apply_time_mask(x)

      if not hasattr(self, "_printed"):
          print("Input shape:", x.shape)
          print("Input stats → mean:", x.mean().item(), "std:", x.std().item(), "min:", x.min().item(), "max:", x.max().item())
          self._printed = True

      return self.whisper_model(input_features=x, labels=labels)

    def _apply_time_mask(self, x, mask_pct=0.075, num_masks=20):
        B, T, D = x.shape
        for b in range(B):
            for _ in range(num_masks):
                t = int(T * mask_pct * torch.rand(1).item())
                t0 = torch.randint(0, T - t + 1, (1,))
                x[b, t0:t0 + t, :] = 0
        return x


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