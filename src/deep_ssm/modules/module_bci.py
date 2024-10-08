import lightning as L
import torch
from deep_ssm.metrics.bci_metrics import calculate_cer

class BCIDecoder(L.LightningModule):
  def __init__(self, args, model):
    super().__init__()
    self.args = args
    self.model = model

    self.loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

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
    self.log(f"ctc_loss_{flag_name}", loss)

    # Compute CER
    if compute_cer:
      train_cer = calculate_cer(pred, input_len, y, y_len)
      self.log(f"cer_{flag_name}", train_cer)
    return loss

  def validation_step(self, batch, batch_idx):
    self._custom_step(batch, batch_idx, flag_name='validation')

  def test_step(self, batch, batch_idx):
    self._custom_step(batch, batch_idx, flag_name='test')

  def configure_optimizers(self):
    if self.args.optimizer_cfg.type == "adam":
      optimizer = torch.optim.Adam(self.parameters(), **self.args.optimizer_cfg.configs)
    else:
      raise NotImplementedError(f"Optimizer {self.args.optimizer_cfg.type} not implemented")
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
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        **self.args.scheduler_cfg.configs,
      )
    elif self.args.scheduler_cfg.type == 'reduce_on_plateau':
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **self.args.scheduler_cfg.configs,
      )
    else:
      raise NotImplementedError(f"Scheduler {self.args.scheduler_cfg.type} not implemented")

    # Copy all scheduler except 'configs' and 'type'
    scheduler_cfg_copy = {k: v for k, v in self.args.scheduler_cfg.items() if k not in ['configs', 'type']}

    scheduler_cfg_copy['scheduler'] = scheduler
    #lr_scheduler_config = { "scheduler" : scheduler}#, "optimizer": optimizer}
    #lr_scheduler_config.update(scheduler_cfg_copy)

    return [optimizer], [scheduler_cfg_copy]


class BCIModule(L.LightningModule):
  """
  Module for BCI model
  Define the loss, train/val/test steps and the optimizer/scheduler.
  """
  def __init__(self, args, model):
    super().__init__()
    self.args = args
    self.model = model


    # lambda_reconstruction:
    if self.model.lambda_cls > 0:
      self.loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    if self.model.lambda_pred > 0:
      if self.args.module_cfg.pred_loss == "mse":
        self.loss_pred = torch.nn.MSELoss(reduction="mean")
      elif self.args.module_cfg.pred_loss == "poisson":
        self.loss_fn = torch.nn.PoissonNLLLoss(reduction="none", log_input=self.args.module_cfg.use_lograte)

  def get_lens(self, X_len):
    if self.model.unfolding:
      input_len = ((X_len - self.model.kernelLen) / self.model.strideLen).to(X_len.dtype)
    else:
      input_len = X_len
    return input_len


  def training_step(self, batch, batch_idx):
    loss = self._custom_step(batch, batch_idx, flag_name='train', compute_cer=True)
    return loss

  def validation_step(self, batch, batch_idx):
    self._custom_step(batch, batch_idx, flag_name='validation')

  def test_step(self, batch, batch_idx):
    self._custom_step(batch, batch_idx, flag_name='test')

  def _custom_step(self, batch, batch_idx, flag_name='test', compute_cer=True):

    X, y, X_len, y_len, dayIdx = batch

    # in case we use unfolding, we need to adjust the input length
    input_len = self.get_lens(X_len)

    # forward pass
    output, pred = self.model(X, dayIdx)

    # calculate loss
    loss = 0

    # CTC loss between classes and inputs
    if self.model.lambda_cls > 0:
      loss_ctc = self.loss_ctc(
        log_probs=torch.permute(pred.log_softmax(2), [1, 0, 2]),
        targets=y,
        input_lengths=input_len,
        target_lengths=y_len,
      )
      self.log(f"ctc_loss_{flag_name}", loss_ctc)
      loss += self.model.lambda_cls * loss_ctc

    # Next step loss
    if self.model.lambda_pred > 0:
      loss_pred = self.loss_pred(output[:,:-1], X[:,1:])
      self.log(f"pred_loss_{flag_name}", loss_pred)
      loss += self.model.lambda_pred * loss_pred

    # Compute CER
    if compute_cer:
      train_cer = calculate_cer(pred, input_len, y, y_len)
      self.log(f"cer_{flag_name}", train_cer)

    return loss

  def configure_optimizers(self):
    if self.args.optimizer_cfg.type == "adam":
      optimizer = torch.optim.Adam(self.parameters(), **self.args.optimizer_cfg.configs)
    else:
      raise NotImplementedError(f"Optimizer {self.args.optimizer_cfg.type} not implemented")
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
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        **self.args.scheduler_cfg.configs,
      )
    elif self.args.scheduler_cfg.type == 'reduce_on_plateau':
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **self.args.scheduler_cfg.configs,
      )
    else:
      raise NotImplementedError(f"Scheduler {self.args.scheduler_cfg.type} not implemented")

    # Copy all scheduler except 'configs' and 'type'
    scheduler_cfg_copy = {k: v for k, v in self.args.scheduler_cfg.items() if k not in ['configs', 'type']}

    scheduler_cfg_copy['scheduler'] = scheduler

    return [optimizer], [scheduler_cfg_copy]





