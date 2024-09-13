import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from safari import utils
import safari.models.nn.utils as U
from safari.dataloaders import SequenceDataset  # TODO make registry
from safari.tasks import decoders, encoders, tasks
from safari.utils import registry
from safari.utils.optim_groups import add_optimizer_hooks
import pytorch_lightning as pl

log = utils.train.get_logger(__name__)


class SequenceLightningModule(pl.LightningModule):
  def __init__(self, config):
    # Disable profiling executor. This reduces memory and increases speed.
    try:
      torch._C._jit_set_profiling_executor(False)
      torch._C._jit_set_profiling_mode(False)
    except AttributeError:
      pass

    super().__init__()
    # Passing in config expands it one level, so can access by self.hparams.train instead of self.hparams.config.train
    self.save_hyperparameters(config, logger=False)

    # Dataset arguments
    self.dataset = SequenceDataset.registry[self.hparams.dataset._name_](
      **self.hparams.dataset
    )

    # Check hparams
    self._check_config()

    # PL has some bugs, so add hooks and make sure they're only called once
    self._has_setup = False

    self.setup()  ## Added by KS

  def setup(self, stage=None):
    if not self.hparams.train.disable_dataset:
      self.dataset.setup()

    # We need to set up the model in setup() because for some reason when training with DDP, one GPU uses much more memory than the others
    # In order to not overwrite the model multiple times during different stages, we need this hack
    # TODO PL 1.5 seems to have an option to skip hooks to avoid this
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/5410#issuecomment-762257024
    if self._has_setup:
      return
    else:
      self._has_setup = True

    # Convenience feature: if model specifies encoder, combine it with main encoder
    encoder_cfg = utils.to_list(self.hparams.encoder) + utils.to_list(
      self.hparams.model.pop("encoder", None)
    )
    decoder_cfg = utils.to_list(
      self.hparams.model.pop("decoder", None)
    ) + utils.to_list(self.hparams.decoder)

    # Instantiate model
    self.model = utils.instantiate(registry.model, self.hparams.model)
    if (name := self.hparams.train.post_init_hook['_name_']) is not None:
      kwargs = self.hparams.train.post_init_hook.copy()
      del kwargs['_name_']
      for module in self.modules():
        if hasattr(module, name):
          getattr(module, name)(**kwargs)

    # Instantiate the task
    self.task = utils.instantiate(
      tasks.registry, self.hparams.task, dataset=self.dataset, model=self.model
    )

    # Create encoders and decoders
    encoder = encoders.instantiate(
      encoder_cfg, dataset=self.dataset, model=self.model
    )
    decoder = decoders.instantiate(
      decoder_cfg, model=self.model, dataset=self.dataset
    )

    # Extract the modules so they show up in the top level parameter count
    self.encoder = U.PassthroughSequential(self.task.encoder, encoder)
    self.decoder = U.PassthroughSequential(decoder, self.task.decoder)
    self.loss = self.task.loss
    self.loss_val = self.task.loss
    if hasattr(self.task, 'loss_val'):
      self.loss_val = self.task.loss_val
    self.metrics = self.task.metrics
    self.train_torchmetrics = self.task.train_torchmetrics
    self.val_torchmetrics = self.task.val_torchmetrics
    self.test_torchmetrics = self.task.test_torchmetrics

  def load_state_dict(self, state_dict, strict=True):
    if self.hparams.train.pretrained_model_state_hook['_name_'] is not None:
      model_state_hook = utils.instantiate(
        registry.model_state_hook,
        self.hparams.train.pretrained_model_state_hook.copy(),
        partial=True,
      )
      # Modify the checkpoint['state_dict'] inside model_state_hook e.g. to inflate 2D convs to 3D convs
      state_dict = model_state_hook(self.model, state_dict)

    print("Custom load_state_dict function is running.")

    # note, it needs to return something from the normal function we overrided
    return super().load_state_dict(state_dict, strict=strict)

  def _check_config(self):
    assert self.hparams.train.state.mode in [None, "none", "null", "reset", "bptt", "tbptt"]
    assert (
      (n := self.hparams.train.state.n_context) is None
      or isinstance(n, int)
      and n >= 0
    )
    assert (
      (n := self.hparams.train.state.n_context_eval) is None
      or isinstance(n, int)
      and n >= 0
    )

  def _initialize_state(self):
    """Called at model setup and start of epoch to completely reset state"""
    self._state = None
    self._memory_chunks = []

  def _reset_state(self, batch, device=None):
    """Called to construct default_state when necessary, e.g. during BPTT"""
    device = device or batch[0].device
    self._state = self.model.default_state(*batch[0].shape[:1], device=device)

  def _detach_state(self, state):
    if isinstance(state, torch.Tensor):
      return state.detach()
    elif isinstance(state, tuple):
      return tuple(self._detach_state(s) for s in state)
    elif isinstance(state, list):
      return [self._detach_state(s) for s in state]
    elif isinstance(state, dict):
      return {k: self._detach_state(v) for k, v in state.items()}
    elif state is None:
      return None
    else:
      raise NotImplementedError

  def _process_state(self, batch, batch_idx, train=True):
    """Handle logic for state context."""
    # Number of context steps
    key = "n_context" if train else "n_context_eval"
    n_context = self.hparams.train.state.get(key)

    # Don't need to do anything if 0 context steps. Make sure there is no state
    if n_context == 0 and self.hparams.train.state.mode not in ['tbptt']:
      self._initialize_state()
      return

    # Reset state if needed
    if self.hparams.train.state.mode == "reset":
      if batch_idx % (n_context + 1) == 0:
        self._reset_state(batch)

    # Pass through memory chunks
    elif self.hparams.train.state.mode == "bptt":
      self._reset_state(batch)
      with torch.no_grad():  # should be unnecessary because individual modules should handle this
        for _batch in self._memory_chunks:
          self.forward(_batch)
      # Prepare for next step
      self._memory_chunks.append(batch)
      self._memory_chunks = self._memory_chunks[-n_context:]

    elif self.hparams.train.state.mode == 'tbptt':
      _, _, z = batch
      reset = z["reset"]
      if reset:
        self._reset_state(batch)
      else:
        self._state = self._detach_state(self._state)

  def forward(self, batch):
    return self.task.forward(batch, self.encoder, self.model, self.decoder, self._state)

  def step(self, x_t):
    x_t, *_ = self.encoder(x_t)  # Potential edge case for encoders that expect (B, L, H)?
    x_t, state = self.model.step(x_t, state=self._state)
    self._state = state
    # x_t = x_t[:, None, ...] # Dummy length
    # x_t, *_ = self.decoder(x_t, state=state)
    # x_t = x_t[:, 0, ...]
    x_t, *_ = self.decoder.step(x_t, state=state)
    return x_t

  def _shared_step(self, batch, batch_idx, prefix="train"):

    self._process_state(batch, batch_idx, train=(prefix == "train"))
    x, y, w = self.forward(batch)

    # Loss
    if prefix == 'train':
      loss = self.loss(x, y, **w)
    else:
      loss = self.loss_val(x, y, **w)

    # Metrics
    metrics = self.metrics(x, y, **w)
    metrics["loss"] = loss
    metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    # Calculate torchmetrics
    torchmetrics = getattr(self, f'{prefix}_torchmetrics')
    torchmetrics(x, y, loss=loss)

    log_on_step = 'eval' in self.hparams and self.hparams.eval.get('log_on_step', False) and prefix == 'train'

    self.log_dict(
      metrics,
      on_step=log_on_step,
      on_epoch=True,
      prog_bar=True,
      add_dataloader_idx=False,
      sync_dist=True,
    )

    # log the whole dict, otherwise lightning takes the mean to reduce it
    # https://pytorch-lightning.readthedocs.io/en/stable/visualize/logging_advanced.html#enable-metrics-for-distributed-training
    self.log_dict(
      torchmetrics,
      on_step=log_on_step,
      on_epoch=True,
      prog_bar=True,
      add_dataloader_idx=False,
      sync_dist=True,
    )
    return loss

  def on_train_epoch_start(self):
    # Reset training torchmetrics
    self.task._reset_torchmetrics("train")

  # def on_train_epoch_end(self, outputs):
  #  pass
    # Log training torchmetrics
    # super().training_epoch_end(outputs)
    # self.log_dict(
    #     {f"train/{k}": v for k, v in self.task.get_torchmetrics("train").items()},
    #     on_step=False,
    #     on_epoch=True,
    #     prog_bar=True,
    #     add_dataloader_idx=False,
    #     sync_dist=True,
    # )

  def on_validation_epoch_start(self):
    # Reset all validation torchmetrics
    for name in self.val_loader_names:
      self.task._reset_torchmetrics(name)

  #def on_validation_epoch_end(self, outputs):
    # Log all validation torchmetrics
    #super().validation_epoch_end(outputs)
  #  pass
    # for name in self.val_loader_names:
    #     self.log_dict(
    #         {f"{name}/{k}": v for k, v in self.task.get_torchmetrics(name).items()},
    #         on_step=False,
    #         on_epoch=True,
    #         prog_bar=True,
    #         add_dataloader_idx=False,
    #         sync_dist=True,
    #     )

  def on_test_epoch_start(self):
    # Reset all test torchmetrics
    for name in self.test_loader_names:
      self.task._reset_torchmetrics(name)

  #def on_test_epoch_end(self):
    # Log all test torchmetrics
    # super().test_epoch_end(outputs)
    # for name in self.test_loader_names:
    #     self.log_dict(
    #         {f"{name}/{k}": v for k, v in self.task.get_torchmetrics(name).items()},
    #         on_step=False,
    #         on_epoch=True,
    #         prog_bar=True,
    #         add_dataloader_idx=False,
    #         sync_dist=True,
    #     )

  def training_step(self, batch, batch_idx, dataloader_idx=0):
    loss = self._shared_step(batch, batch_idx, prefix="train")

    # Log the loss explicitly so it shows up in WandB
    # Note that this currently runs into a bug in the progress bar with ddp (as of 1.4.6)
    # https://github.com/PyTorchLightning/pytorch-lightning/pull/9142
    # We additionally log the epochs under 'trainer' to get a consistent prefix with 'global_step'
    loss_epoch = {"trainer/loss": loss, "trainer/epoch": self.current_epoch}
    self.log_dict(
      loss_epoch,
      on_step=True,
      on_epoch=False,
      prog_bar=False,
      add_dataloader_idx=False,
      sync_dist=True,
    )

    # Log any extra info that the models want to expose (e.g. output norms)
    metrics = {}
    for module in list(self.modules())[1:]:
      if hasattr(module, "metrics"):
        metrics.update(module.metrics)

    self.log_dict(
      metrics,
      on_step=True,
      on_epoch=False,
      prog_bar=False,
      add_dataloader_idx=False,
      sync_dist=True,
    )

    return loss

  def validation_step(self, batch, batch_idx, dataloader_idx=0):
    ema = (
      self.val_loader_names[dataloader_idx].endswith("/ema")
      and self.optimizers().optimizer.stepped
    )  # There's a bit of an annoying edge case with the first (0-th) epoch; it has to be excluded due to the initial sanity check
    if ema:
      self.optimizers().swap_ema()
    loss = self._shared_step(
      batch, batch_idx, prefix=self.val_loader_names[dataloader_idx]
    )
    if ema:
      self.optimizers().swap_ema()

    return loss

  def test_step(self, batch, batch_idx, dataloader_idx=0):
    return self._shared_step(
      batch, batch_idx, prefix=self.test_loader_names[dataloader_idx]
    )

  def configure_optimizers(self):
    # Set zero weight decay for some params
    if 'optimizer_param_grouping' in self.hparams.train:
      add_optimizer_hooks(self.model, **self.hparams.train.optimizer_param_grouping)

    # Normal parameters
    all_params = list(self.parameters())
    params = [p for p in all_params if not hasattr(p, "_optim")]

    optimizer = utils.instantiate(registry.optimizer, self.hparams.optimizer, params)

    del self.hparams.optimizer._name_

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
    hps = [
      # dict(s) for s in set(frozenset(hp.items()) for hp in hps)
      dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
      # dict(s) for s in dict.fromkeys(frozenset(hp.items()) for hp in hps)
    ]  # Unique dicts
    print("Hyperparameter groups", hps)
    for hp in hps:
      params = [p for p in all_params if getattr(p, "_optim", None) == hp]
      optimizer.add_param_group(
        {"params": params, **self.hparams.optimizer, **hp}
      )

    ### Layer Decay ###

    if self.hparams.train.layer_decay['_name_'] is not None:
      get_num_layer = utils.instantiate(
        registry.layer_decay,
        self.hparams.train.layer_decay['_name_'],
        partial=True,
      )

      # Go through all parameters and get num layer
      layer_wise_groups = {}
      num_max_layers = 0
      for name, p in self.named_parameters():
        # Get layer id for each parameter in the model
        layer_id = get_num_layer(name)

        # Add to layer wise group
        if layer_id not in layer_wise_groups:
          layer_wise_groups[layer_id] = {
            'params': [],
            'lr': None,
            'weight_decay': self.hparams.optimizer.weight_decay
          }
        layer_wise_groups[layer_id]['params'].append(p)

        if layer_id > num_max_layers: num_max_layers = layer_id

      # Update lr for each layer
      for layer_id, group in layer_wise_groups.items():
        group['lr'] = self.hparams.optimizer.lr * (self.hparams.train.layer_decay.decay ** (num_max_layers - layer_id))

      # Reset the torch optimizer's param groups
      optimizer.param_groups = []
      for layer_id, group in layer_wise_groups.items():
        optimizer.add_param_group(group)

    # Print optimizer info for debugging
    keys = set([k for hp in hps for k in hp.keys()])  # Special hparams
    utils.train.log_optimizer(log, optimizer, keys)
    # Configure scheduler
    if "scheduler" not in self.hparams:
      return optimizer
    lr_scheduler = utils.instantiate(
      registry.scheduler, self.hparams.scheduler, optimizer
    )
    scheduler = {
      "scheduler": lr_scheduler,
      "interval": self.hparams.train.interval,  # 'epoch' or 'step'
      "monitor": self.hparams.train.monitor,
      "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
    }
    # See documentation for how to configure the return
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
    return [optimizer], [scheduler]

  def train_dataloader(self):
    return self.dataset.train_dataloader(**self.hparams.loader)

  def _eval_dataloaders_names(self, loaders, prefix):
    """Process loaders into a list of names and loaders"""
    if utils.is_dict(loaders):
      return [
        f"{prefix}/{k}" if k is not None else prefix for k in loaders.keys()
      ], list(loaders.values())
    elif utils.is_list(loaders):
      return [f"{prefix}/{i}" for i in range(len(loaders))], loaders
    else:
      return [prefix], [loaders]

  def _eval_dataloaders(self):
    # Return all val + test loaders
    val_loaders = self.dataset.val_dataloader(**self.hparams.loader)
    test_loaders = self.dataset.test_dataloader(**self.hparams.loader)
    val_loader_names, val_loaders = self._eval_dataloaders_names(val_loaders, "val")
    test_loader_names, test_loaders = self._eval_dataloaders_names(
      test_loaders, "test"
    )

    # Duplicate datasets for ema
    if self.hparams.train.ema > 0.0:
      val_loader_names += [name + "/ema" for name in val_loader_names]
      val_loaders = val_loaders + val_loaders
      test_loader_names += [name + "/ema" for name in test_loader_names]
      test_loaders = test_loaders + test_loaders

    # adding option to only have val loader at eval (eg if test is duplicate)
    if self.hparams.train.get("remove_test_loader_in_eval", None) is not None:
      return val_loader_names, val_loaders
    # default behavior is to add test loaders in eval
    else:
      return val_loader_names + test_loader_names, val_loaders + test_loaders

  def val_dataloader(self):
    val_loader_names, val_loaders = self._eval_dataloaders()
    self.val_loader_names = val_loader_names
    return val_loaders

  def test_dataloader(self):
    test_loader_names, test_loaders = self._eval_dataloaders()
    self.test_loader_names = ["final/" + name for name in test_loader_names]
    return test_loaders
