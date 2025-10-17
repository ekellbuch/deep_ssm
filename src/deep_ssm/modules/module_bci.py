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
      # ---------- OPTIMIZER ----------
      if self.args.optimizer_cfg.type == "adam":
          optimizer = torch.optim.Adam(self.parameters(), **self.args.optimizer_cfg.configs)
      elif self.args.optimizer_cfg.type == "adamw":
          optimizer = torch.optim.AdamW(self.parameters(), **self.args.optimizer_cfg.configs)
      elif self.args.optimizer_cfg.type == "soap":
          from heavyball import SOAP
          optimizer = SOAP(self.parameters(), **self.args.optimizer_cfg.configs)
      else:
          raise NotImplementedError(f"Optimizer {self.args.optimizer_cfg.type} not implemented")

      # ---------- MAIN SCHEDULER ----------
      scheduler_type = self.args.scheduler_cfg.type
      scheduler_configs = dict(self.args.scheduler_cfg.configs)  # copy to mutate safely
      interval = self.args.scheduler_cfg.get("interval", "epoch")
      frequency = self.args.scheduler_cfg.get("frequency", 1)
      monitor = self.args.scheduler_cfg.get("monitor", "ctc_loss_validation")

      # We’ll create the "main" scheduler first; warmup will wrap it via SequentialLR if requested.
      def make_main_scheduler():
          if scheduler_type == "linear":
              return torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_configs)
          elif scheduler_type == "stepLR":
              return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_configs)
          elif scheduler_type == "cosine_annealing":
              # If interval=="step" and T_max not provided, compute from remaining steps after warmup.
              return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_configs)
          elif scheduler_type == "reduce_on_plateau":
              return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_configs)
          elif scheduler_type == "cosine_annealing_warm_restarts":
              return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_configs)
          elif scheduler_type == "cyclic_lr":
              return torch.optim.lr_scheduler.CyclicLR(optimizer, **scheduler_configs)
          elif scheduler_type == "multistep":
              return torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_configs)
          else:
              raise NotImplementedError(f"Scheduler {scheduler_type} not implemented")

      # ---------- WARMUP (epoch- or step-based) ----------
      # Choose ONE of:
      #  - warmup_epochs (int)       -> epoch-based warmup (keeps your current behavior)
      #  - warmup_steps (int)        -> step-based warmup
      #  - warmup_frac (0..1)        -> step-based warmup as a fraction of total steps
      warmup_epochs = self.args.scheduler_cfg.get("warmup_epochs", 0)
      warmup_steps  = self.args.scheduler_cfg.get("warmup_steps", 0)
      warmup_frac   = self.args.scheduler_cfg.get("warmup_frac", 0.0)

      # Compute total/remaining steps if we are step-based
      total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
      if total_steps is None or total_steps == 0:
          # fallback (e.g., fast_dev_run); doesn’t need to be exact
          total_steps = self.trainer.max_epochs * max(1, getattr(self.trainer, "num_training_batches", 1))

      if warmup_steps == 0 and warmup_frac > 0:
          warmup_steps = max(1, int(warmup_frac * total_steps))

      use_step_warmup = (interval == "step") and (warmup_steps > 0)
      use_epoch_warmup = (interval == "epoch") and (warmup_epochs > 0)

      if use_step_warmup:
          # LinearLR truly ramps each step. Start small but non-zero.
          scheduler_warm = torch.optim.lr_scheduler.LinearLR(
              optimizer,
              start_factor=1e-3,  # start at 0.1% of base LR; make 1e-2 if ramp is too slow
              end_factor=1.0,
              total_iters=warmup_steps,
          )

          # If cosine_annealing and T_max not provided, set it to the remaining steps
          if scheduler_type == "cosine_annealing" and "T_max" not in scheduler_configs:
              scheduler_configs["T_max"] = max(1, total_steps - warmup_steps)
          main_scheduler = make_main_scheduler()

          scheduler = torch.optim.lr_scheduler.SequentialLR(
              optimizer,
              schedulers=[scheduler_warm, main_scheduler],
              milestones=[warmup_steps],
          )
          milestones_for_return = warmup_steps  # just for clarity below

      elif use_epoch_warmup:
          # Epoch-based warmup: scale LR linearly over first warmup_epochs
          # Start at a tiny factor on epoch 0 to avoid zero LR in the first epoch.
          def warmup_lambda(epoch):
              if epoch < warmup_epochs:
                  # from ~0 to 1 over warmup_epochs
                  return max(1e-3, float(epoch + 1) / float(warmup_epochs))
              return 1.0

          scheduler_warm = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

          main_scheduler = make_main_scheduler()
          scheduler = torch.optim.lr_scheduler.SequentialLR(
              optimizer,
              schedulers=[scheduler_warm, main_scheduler],
              milestones=[warmup_epochs],
          )
          milestones_for_return = warmup_epochs
      else:
          # No warmup; just the main scheduler
          # If cosine_annealing with step interval and no T_max, set it now.
          if interval == "step" and scheduler_type == "cosine_annealing" and "T_max" not in scheduler_configs:
              scheduler_configs["T_max"] = total_steps
          scheduler = make_main_scheduler()
          milestones_for_return = None

      # ---------- RETURN FOR LIGHTNING ----------
      scheduler_dict = {
          "scheduler": scheduler,
          "interval": interval,        # "step" for step-wise schedules, "epoch" otherwise
          "frequency": frequency,
      }
      if scheduler_type == "reduce_on_plateau":
          scheduler_dict["monitor"] = monitor

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