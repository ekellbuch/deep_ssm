import lightning as L
import torch
import torch.nn.functional as F
import wandb

class StateTrackingDecoder(L.LightningModule):
  def __init__(self, args, model):
    super().__init__()
    self.args = args
    self.model = model

    self.loss_fn = torch.nn.CrossEntropyLoss()

    self.train_step_outputs = []
    self.validation_step_outputs = []
    self.test_step_outputs = []

  def check_rescaling_factor(self):
    rescale_type = self.args.get("rescale_loss", None)
    if rescale_type == "num_iters":
      # Loss is scaled by the # iterations: steps which run for more iteration get higher weight
      rescale_factor = self.model.model.num_iters
    elif rescale_type == "progressive":
      current_epoch = self.trainer.current_epoch
      raise NotImplementedError("Progressive rescaling not implemented.")
      # rescale_factor = self.model.model.num_iters
    else:
      # Loss is not rescaled: all steps are weighted equally
      rescale_factor = 1
    return rescale_factor

  def _custom_step(self, batch, batch_idx, flag_name='test'):
    src, tgt, mask = batch

    # Forward pass
    logits = self.model(src)  # Shape: [batch_size, seq_len, vocab_size]
    # Reshape for cross entropy
    flat_logits = logits.view(-1, logits.size(-1))  # [batch_size*seq_len, vocab_size]
    flat_targets = tgt.view(-1)  # [batch_size*seq_len]
    flat_mask = mask.view(-1)  # [batch_size*seq_len]

    rescale_factor = self.check_rescaling_factor()
    # Calculate loss only on non-padded positions
    # rescale loss is number of iters
    loss = rescale_factor * self.loss_fn(flat_logits[flat_mask], flat_targets[flat_mask])

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
    return acc

  def _custom_step_final_position(self, batch, batch_idx, flag_name='test'):
    src, tgt, mask = batch

    # Forward pass
    logits = self.model(src)  # Shape: [batch_size, seq_len, vocab_size]

    # Reshape for cross entropy
    flat_logits = logits[:, -1, :].view(-1, logits.size(-1))  # [batch_size*seq_len, vocab_size]
    flat_targets = tgt[:, -1].view(-1)  # [batch_size*seq_len]
    flat_mask = mask[:, -1].view(-1)  # [batch_size*seq_len]

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

  def training_step(self, batch, batch_idx):
    output = self._custom_step(batch, batch_idx, flag_name='train')
    self.train_step_outputs.append(output)
    return output
  
  def on_train_epoch_end(self):
    outputs = self.train_step_outputs
    self._custom_epoch_end(outputs, flag_name='train')
    self.train_step_outputs.clear()
    # wandb.log({"A_norm": torch.linalg.norm(self.model.model.forward_cells[0].A, ord='fro')})

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # x0 = torch.zeros(1, self.model.model.input_size, requires_grad=True).to(device)
    # h0 = torch.zeros(1, self.model.model.hidden_size, requires_grad=True).to(device)

    # # Get the Jacobian
    # jacobian = torch.autograd.functional.jacobian(lambda h: self.model.model.forward_cells[0].original_cell(x0, h), h0)

    # # Reshape Jacobian to (hidden_size, hidden_size)
    # J_f = jacobian.squeeze()
    # eigenvalues, P = torch.linalg.eig(J_f)
    # self.model.model.forward_cells[0].P = P.to(device)

  def validation_step(self, batch, batch_idx):
    output = self._custom_step_final_position(batch, batch_idx, flag_name='validation')
    self.validation_step_outputs.append(output)
    return output

  def on_validation_epoch_end(self):
    outputs = self.validation_step_outputs
    self._custom_epoch_end(outputs, flag_name='validation')
    self.validation_step_outputs.clear()

  def test_step(self, batch, batch_idx):
    output = self._custom_step_final_position(batch, batch_idx, flag_name='test')
    self.test_step_outputs.append(output)
    return output

  def on_test_epoch_end(self):
    outputs = self.test_step_outputs
    test_acc = self._custom_epoch_end(outputs, flag_name='test')
    self.test_step_outputs.clear()
    return test_acc

  def configure_optimizers(self):
    if self.args.optimizer_cfg.type == "adam":
      optimizer = torch.optim.Adam(self.parameters(), **self.args.optimizer_cfg.configs)
    elif self.args.optimizer_cfg.type == "adamw":
      optimizer = torch.optim.AdamW(self.parameters(), **self.args.optimizer_cfg.configs)
    else:
      raise NotImplementedError(f"Optimizer {self.args.optimizer_cfg.type} not implemented")

    # Check if scheduler is not defined
    if not self.args.get("scheduler_cfg", None):
        return optimizer

    # Define Warm-Up Scheduler
    warmup_epochs = self.args.scheduler_cfg.get("warmup_epochs", 0)
    num_steps_per_epoch = self.trainer.estimated_stepping_batches * warmup_epochs

    def warmup_lambda(epoch):
      # Warm-up function: linearly increase LR for first 'warmup_epochs' epochs
      if epoch < warmup_epochs:
        return float(epoch) / float(warmup_epochs)
      else:
        return 1.0  # After warm-up, learning rate stays constant

    # Warm-Up Scheduler
    if warmup_epochs > 0:
      warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
          optimizer,
          start_factor=0.01,
          end_factor=1.0,
          total_iters=num_steps_per_epoch)

    if self.args.scheduler_cfg.type == "linear":
      scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        **self.args.scheduler_cfg.configs
      )
    elif self.args.scheduler_cfg.type == "stepLR":
      scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        **self.args.scheduler_cfg.configs,
      )
    elif self.args.scheduler_cfg.type == "cosine_annealing":
      cosine_params = self.args.scheduler_cfg.configs
      cosine_params['T_max'] = num_steps_per_epoch * cosine_params['T_max']
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        **cosine_params,
      )
    elif self.args.scheduler_cfg.type == 'reduce_on_plateau':
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **self.args.scheduler_cfg.configs,
      )
    elif self.args.scheduler_cfg.type == 'cosine_annealing_warm_restarts':
      scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        **self.args.scheduler_cfg.configs,
      )
    elif self.args.scheduler_cfg.type == 'cyclic_lr':
      scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        **self.args.scheduler_cfg.configs,
      )
    else:
        raise NotImplementedError(f"Scheduler {self.args.scheduler_cfg.type} not implemented")
    # Copy all scheduler except 'configs' and 'type'
    scheduler_cfg_copy = {k: v for k, v in self.args.scheduler_cfg.items() if k not in ['configs', 'type']}

    # Add warmup_scheduler to the scheduler_cfg_copy
    if warmup_epochs > 0:
      # Chained scheduler, where warmup runs first, then the main scheduler
      chained_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        schedulers=[warmup_scheduler, scheduler],
      )
      scheduler_cfg_copy['scheduler'] = chained_scheduler
    else:
      scheduler_cfg_copy['scheduler'] = scheduler
    #lr_scheduler_config = { "scheduler" : scheduler}#, "optimizer": optimizer}
    #lr_scheduler_config.update(scheduler_cfg_copy)

    return [optimizer], [scheduler_cfg_copy]
