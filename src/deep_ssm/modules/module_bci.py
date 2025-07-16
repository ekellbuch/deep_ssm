import lightning as L
import torch
from deep_ssm.metrics.bci_metrics import calculate_cer

class BCIDecoder(L.LightningModule):
  def __init__(self, args, model):
    super().__init__()
    self.args = args
    self.model = model

    self.loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    self.calculate_cer = calculate_cer

  def get_lens(self, X_len):
    if self.model.unfolding:
      input_len = ((X_len - self.model.kernelLen) / self.model.strideLen).to(X_len.dtype)
    else:
      input_len = X_len
    return input_len


  def training_step(self, batch, batch_idx):
    loss = self._custom_step(batch, batch_idx, flag_name='train', compute_cer=False)
    return loss

  def _custom_step(self, batch, batch_idx, flag_name='test', compute_cer=True):

    X, y, X_len, y_len, dayIdx = batch

    input_len = self.get_lens(X_len)

    # forward pass
    pred = self.model(X, dayIdx)

    # calculate CTC loss
    loss = self.loss_ctc(
      log_probs=torch.permute(pred.log_softmax(2), [1, 0, 2]),
      targets=y,
      input_lengths=input_len,
      target_lengths=y_len,
    )
    self.log(f"ctc_loss_{flag_name}", loss, on_epoch=True)

    # Compute CER
    if compute_cer:
      train_cer = self.calculate_cer(pred, input_len, y, y_len)
      self.log(f"cer_{flag_name}", train_cer, on_epoch=True)
    return loss

  def validation_step(self, batch, batch_idx):
    self._custom_step(batch, batch_idx, flag_name='validation', compute_cer=False)

  def test_step(self, batch, batch_idx):
    self._custom_step(batch, batch_idx, flag_name='test')



  def configure_optimizers(self):
    if self.args.optimizer_cfg.type == "adam":
      optimizer = torch.optim.Adam(self.parameters(), **self.args.optimizer_cfg.configs)
    elif self.args.optimizer_cfg.type == "adamw":
      optimizer = torch.optim.AdamW(self.parameters(), **self.args.optimizer_cfg.configs)
    else:
      raise NotImplementedError(f"Optimizer {self.args.optimizer_cfg.type} not implemented")

    # Define Warm-Up Scheduler
    warmup_epochs = self.args.scheduler_cfg.get("warmup_epochs", 0)

    def warmup_lambda(epoch):
      # Warm-up function: linearly increase LR for first 'warmup_epochs' epochs
      if epoch < warmup_epochs:
        return float(epoch) / float(warmup_epochs)
      else:
        return 1.0  # After warm-up, learning rate stays constant

    # Warm-Up Scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lr_lambda=warmup_lambda)

    # Main scheduler
    scheduler_type = self.args.scheduler_cfg.type
    scheduler_configs = self.args.scheduler_cfg.configs

    if scheduler_type == "linear":
      main_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        **scheduler_configs
      )
    elif scheduler_type == "stepLR":
      main_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        **scheduler_configs,
      )
    elif scheduler_type == "cosine_annealing":
      main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        **scheduler_configs,
      )
    elif scheduler_type == 'reduce_on_plateau':
      main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **scheduler_configs,
      )
    elif scheduler_type == 'cosine_annealing_warm_restarts':
      main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        **scheduler_configs,
      )
    elif scheduler_type == 'cyclic_lr':
      main_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        **scheduler_configs,
      )
    elif scheduler_type == 'multistep':
      main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        **scheduler_configs,
      )
    else:
      raise NotImplementedError(f"Scheduler {scheduler_type} not implemented")

    # Copy all scheduler except 'configs' and 'type'
    scheduler_cfg_copy = {k: v for k, v in self.args.scheduler_cfg.items() if k not in ['configs', 'type']}

    # Wrap with SequentialLR if warmup is used
    if warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = main_scheduler

    # Lightning expects these fields
    scheduler_dict = {
        "scheduler": scheduler,
        "interval": self.args.scheduler_cfg.get("interval", "epoch"),
        "frequency": self.args.scheduler_cfg.get("frequency", 1),
    }

    # Only add 'monitor' if needed
    if scheduler_type == "reduce_on_plateau":
        scheduler_dict["monitor"] = self.args.scheduler_cfg.get("monitor", "ctc_loss_validation")

    return [optimizer], [scheduler_dict]
  

class BCIEncoder(BCIDecoder):
  def __init__(self, args, model):
    super().__init__(args, model)

  def get_lens(self, X_len):
    return self.model.bit_model.compute_length(X_len)

  def _custom_step(self, batch, batch_idx, flag_name='test', compute_cer=True):

    X, y, X_len, y_len, dayIdx = batch

    input_len = self.get_lens(X_len)

    # forward pass
    pred = self.model(X, X_len, dayIdx)

    # calculate CTC loss
    loss = self.loss_ctc(
      log_probs=torch.permute(pred.log_softmax(2), [1, 0, 2]),
      targets=y,
      input_lengths=input_len,
      target_lengths=y_len,
    )
    self.log(f"ctc_loss_{flag_name}", loss, on_epoch=True)

    #if flag_name == 'train':
    #  print(f"train loss: {loss}")
    # Compute CER
    if compute_cer:
      train_cer = self.calculate_cer(pred, input_len, y, y_len)
      self.log(f"cer_{flag_name}", train_cer, on_epoch=True)
    return loss