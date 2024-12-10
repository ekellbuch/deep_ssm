import lightning as L
import torch
from deep_ssm.metrics.bci_metrics import calculate_cer

class ParityDecoder(L.LightningModule):
  def __init__(self, args, model):
    super().__init__()
    self.args = args
    self.model = model

    self.loss_fn = torch.nn.CrossEntropyLoss()

    self.train_step_outputs = []
    self.validation_step_outputs = []
    self.test_step_outputs = []

  def _custom_step(self, batch, batch_idx, flag_name='test'):
    src, tgt, mask = batch

    # Forward pass
    logits = self.model(src)  # Shape: [batch_size, seq_len, vocab_size]

    # Reshape for cross entropy
    flat_logits = logits.view(-1, logits.size(-1))  # [batch_size*seq_len, vocab_size]
    flat_targets = tgt.view(-1)  # [batch_size*seq_len]
    flat_mask = mask.view(-1)  # [batch_size*seq_len]

    # Calculate loss only on non-padded positions
    loss = self.loss_fn(flat_logits[flat_mask], flat_targets[flat_mask])
    # self.log(f"loss_{flag_name}", loss)

    # Calculate accuracy on non-padded positions
    predictions = flat_logits.argmax(dim=-1)
    correct_predictions = ((predictions == flat_targets) & flat_mask).sum().item()
    total_tokens = flat_mask.sum().item()
    total_loss = loss.item() * flat_mask.sum().item()

    return {"loss": loss, "correct_predictions": correct_predictions, 
            "total_tokens": total_tokens, "total_loss": total_loss}
  
  def _custom_epoch_end(self, outputs, flag_name='train'):
    total_tokens = sum([output["total_tokens"] for output in outputs])
    total_loss = sum([output["total_loss"] for output in outputs])
    avg_loss = total_loss / total_tokens
    correct_predictions = sum([output["correct_predictions"] for output in outputs])
    acc = correct_predictions / total_tokens
    self.log(f"avg_{flag_name}_loss", avg_loss, on_epoch=True, prog_bar=True)
    self.log(f"avg_{flag_name}_acc", acc, on_epoch=True, prog_bar=True)

  def training_step(self, batch, batch_idx):
    output = self._custom_step(batch, batch_idx, flag_name='train')
    self.train_step_outputs.append(output)
    return output
  
  def on_train_epoch_end(self):
    outputs = self.train_step_outputs
    self._custom_epoch_end(outputs, flag_name='train')
    self.train_step_outputs.clear()

  def validation_step(self, batch, batch_idx):
    output = self._custom_step(batch, batch_idx, flag_name='validation')
    self.validation_step_outputs.append(output)
    return output

  def on_validation_epoch_end(self):
    outputs = self.validation_step_outputs
    self._custom_epoch_end(outputs, flag_name='validation')
    self.validation_step_outputs.clear()

  def test_step(self, batch, batch_idx):
    output = self._custom_step(batch, batch_idx, flag_name='test')
    self.test_step_outputs.append(output)
    return output

  def on_test_epoch_end(self):
    outputs = self.test_step_outputs
    self._custom_epoch_end(outputs, flag_name='test')
    self.test_step_outputs.clear()

  def configure_optimizers(self):
    if self.args.optimizer_cfg.type == "adam":
      optimizer = torch.optim.Adam(self.parameters(), **self.args.optimizer_cfg.configs)
    elif self.args.optimizer_cfg.type == "adamw":
      optimizer = torch.optim.AdamW(self.parameters(), **self.args.optimizer_cfg.configs)
    else:
      raise NotImplementedError(f"Optimizer {self.args.optimizer_cfg.type} not implemented")
    return optimizer
