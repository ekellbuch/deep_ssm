import lightning as L
import torch
from deep_ssm.metrics.bci_metrics import calculate_cer
from torchmetrics.text import WordErrorRate


class BaseBCIModule(L.LightningModule):
  def __init__(self, args):
    super().__init__()
    self.args = args

  def configure_optimizers(self):
    if self.args.optimizer_cfg.type == "adam":
      optimizer = torch.optim.Adam(self.parameters(), **self.args.optimizer_cfg.configs)
    elif self.args.optimizer_cfg.type == "adamw":
      optimizer = torch.optim.AdamW(self.parameters(), **self.args.optimizer_cfg.configs)
    else:
      raise NotImplementedError(f"Optimizer {self.args.optimizer_cfg.type} not implemented")

    # Define Warm-Up Scheduler
    warmup_epochs = self.args.scheduler_cfg.get("warmup_epochs", 0)
    n_steps = self.trainer.estimated_stepping_batches
    if warmup_epochs > 0:
      n_warmup_steps = warmup_epochs*n_steps

      warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                          start_factor=0.1,
                                                          end_factor=1.0,
                                                          total_iters=n_warmup_steps,
                                                          )

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
      scbeduler_configs = self.args.scheduler_cfg.configs
      scbeduler_configs['T_max'] =scbeduler_configs['T_max']*n_steps
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        **scbeduler_configs,
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
      chained_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, scheduler],
        milestones=[warmup_epochs]
      )
      scheduler_cfg_copy['scheduler'] = chained_scheduler
    else:
      scheduler_cfg_copy['scheduler'] = scheduler
    #lr_scheduler_config = { "scheduler" : scheduler}#, "optimizer": optimizer}
    #lr_scheduler_config.update(scheduler_cfg_copy)
    return [optimizer], [scheduler_cfg_copy]


class BCIDecoder(BaseBCIModule):
  def __init__(self, args, model):
    super().__init__(args)
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
    self._custom_step(batch, batch_idx, flag_name='validation', compute_cer=False)

  def test_step(self, batch, batch_idx):
    self._custom_step(batch, batch_idx, flag_name='test')


class BCIWhisperModule(BaseBCIModule):
  def __init__(self, args, model, processor):
    super().__init__(args)
    self.model = model
    self.processor = processor
    self.wer = WordErrorRate()

  def training_step(self, batch, batch_idx):
    loss = self._custom_step(batch, batch_idx, flag_name='train', compute_wer=True)
    return loss

  def _custom_step(self, batch, batch_idx, flag_name='train', compute_wer=False):
    X, y, X_len, y_len, dayIdx, transcriptions = batch

    # forward pass
    outputs = self.model(X, labels=transcriptions)

    # calculate loss
    loss = outputs.loss
    self.log(f"loss_{flag_name}", loss)

    # Compute WER
    if compute_wer:
      predictions = self.processor.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
      references = self.processor.batch_decode(transcriptions, skip_special_tokens=True)
      #print(f"{flag_name} True sequence: ", references)
      #print(f"{flag_name} Prediction: ", predictions)
      batch_wer = self.wer(predictions,references)
      self.log(f"wer_{flag_name}", batch_wer)
    return loss

  def validation_step(self, batch, batch_idx):
    _ = self._custom_step(batch, batch_idx, flag_name='validation', compute_wer=True)
    return

  def test_step(self, batch, batch_idx):
    _ = self._custom_step(batch, batch_idx, flag_name='test', compute_wer=True)
    return