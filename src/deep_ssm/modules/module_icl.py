import lightning as L
import torch


class ICLRegression(L.LightningModule):
  def __init__(self, args, model):
    super().__init__()
    self.args = args
    self.model = model

    self.train_loss = torch.nn.MSELoss()
    self.test_loss = torch.nn.MSELoss(reduction="none")

  def training_step(self, batch, batch_idx):
    loss = self._custom_step(batch, batch_idx, flag_name='train')
    return loss

  def _custom_step(self, batch, batch_idx, flag_name="train"):

    x, y = batch
    # forward pass
    pred = self.model(x)
    # BLD
    loss = self.train_loss(pred[:,0::2,0], y[:,0::2,0])
    self.log(f"loss_{flag_name}", loss)

    return loss

  def validation_step(self, batch, batch_idx):
    self._custom_step(batch, batch_idx, flag_name='validation') # xg switch to true to look at CER on val

  def test_step(self, batch, batch_idx):
    x, y = batch
    # forward pass
    pred = self.model(x)

    # BDL
    loss = self.test_loss(pred[:,0::2,0], y[:,0::2,0]).mean(0)
    for seq_len in range(len(loss)):
      self.log(f"loss_test_{seq_len}", loss[seq_len])
    return loss

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
