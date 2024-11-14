"""
Each array captures signals at a sampling frequency of 30kHz
The neural signal were binned into 20 millisecond intervals
so we ended up with a rate of 50 Hz (bins) per second.

Memory, it doesn't fit with batch_size 64 but it does fit with batch_size 32
Version which works, now try with lightning
"""
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

torch.autograd.set_detect_anomaly(True)

TOY_CONFIG = "/home/groups/swl1/ekb/Projects/deepseq/deep_ssm/configs/bci/baseline_whisper.yaml"



def print_trainable_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameter: {name}, Shape: {param.shape}")

# Helper function to calculate trainable parameters
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# read baseline_mamba.yaml
def create_cfg() -> dict:
    """Load all toy data config file without hydra."""
    cfg = yaml.load(open(str(TOY_CONFIG)), Loader=yaml.FullLoader)
    return OmegaConf.create(cfg)


def main():

    # Load config file:
    args = create_cfg()
    L.seed_everything(args.seed)

    # Load dataset
    datamodule = SpeechDataModule(args.data_cfg)
    datamodule.setup()
    #batch =  next(iter(datamodule.train_dataloader()))

    spike_feature_dim = 256  # Example feature dimension for neural spike data

    # Load Whisper model and processor
    model_name = "openai/whisper-small.en"
    # extracts mel-filter bank features from raw speech using Short time Fourier Transform
    #feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    # loads whisper model
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    #test_input = torch.randn( 80, 3000).to(device)  # Adjust to match your batch and input size
    #model = model.to(device)
    # summary(model, (3000,))
    
    # Modify the model input layer to accept neural data
    class ModifiedWhisper(nn.Module):

        def __init__(self,whisper_model):
            super(ModifiedWhisper, self).__init__()
            self.input_layer = nn.Conv1d(in_channels=spike_feature_dim, out_channels=80, kernel_size=3, stride=1, padding=1)
            #self.feature_extractor = feature_extractor#(feature_size=spike_feature_dim)
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

    # Initialize modified model
    modified_model = ModifiedWhisper(model)

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
    print(f"Total trainable parameters: {count_trainable_params(modified_model)}")



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
        print(f"Total trainable parameters after LoRA: {count_trainable_params(modified_model)}")

    # Define training parameters
    modified_model.to(device)

    optimizer = optim.AdamW(modified_model.parameters(), lr=1e-3)


    # Warm-up and total training steps
    warmup_steps = 10
    total_steps = 100  #len(dataloader) * epochs

    # Lambda function for learning rate
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    # Scheduler with warm-up and decay
    #scheduler = LambdaLR(optimizer, lr_lambda)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-8)

    wer_metric = load_metric("wer")

    # Create DataLoader
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()

    # Training loop
    num_epochs = 5
    modified_model.train()
    #for epoch in tqdm(range(num_epochs)):
    batches = next(iter(train_loader))
    batches = [batches]

    for epoch in range(num_epochs):
        total_loss = 0
        wer_sum = 0
        total_samples = 0

        # Overfit one batch:
        #batches = [train_loader]
        #for batch in tqdm(batches):
        for batch in batches:
            spike_features = batch[0].to(device)  # batch x length x num_features
            labels = batch[-1].to(device)  # [batch, seq_len]
            # Forward pass
            outputs = modified_model(spike_features, labels=labels)

            #breakpoint()
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            total_loss += loss.item()
            # Add gradient clipping here
            torch.nn.utils.clip_grad_norm_(modified_model.parameters(), max_norm=1.0)
 
            optimizer.step()

            # Decode and calculate WER
            predictions = processor.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
            references = processor.batch_decode(labels, skip_special_tokens=True)
            batch_wer = wer_metric.compute(predictions=predictions, references=references)

            print("True sequence: ", references)
            print("Prediction: ", predictions)
            wer_sum += batch_wer * spike_features.size(0)
            total_samples += spike_features.size(0)
        avg_loss = total_loss / len(train_loader)
        avg_wer = wer_sum / total_samples

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train WER: {avg_wer:.4f}")

        # Update the scheduler at the end of each epoch
        scheduler.step(avg_loss)

        # calculate test time 
    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
