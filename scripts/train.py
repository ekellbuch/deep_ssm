"""

"""
from typing import Callable, List, Sequence

import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf

import safari.utils as utils
import safari.utils.train
from safari.utils import registry

from deep_ssm.module import SequenceLightningModule
from safari.logging import CustomWandbLogger

log = safari.utils.train.get_logger(__name__)

# Turn on TensorFloat32 (speeds up large model training substantially)
import torch.backends

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)


### pytorch-lightning utils and entrypoint ###

def create_trainer(config, **kwargs):
  callbacks: List[pl.Callback] = []
  logger = None

  # WandB Logging
  if config.get("wandb") is not None:
    # Pass in wandb.init(config=) argument to get the nice 'x.y.0.z' hparams logged
    # Can pass in config_exclude_keys='wandb' to remove certain groups
    import wandb

    logger = CustomWandbLogger(
      config=utils.to_dict(config, recursive=True),
      settings=wandb.Settings(start_method="fork"),
      **config.wandb,
    )

  # Lightning callbacks
  if "callbacks" in config:
    for _name_, callback in config.callbacks.items():
      if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
        continue
      log.info(f"Instantiating callback <{registry.callbacks[_name_]}>")
      callback._name_ = _name_
      callbacks.append(utils.instantiate(registry.callbacks, callback))

  # Add ProgressiveResizing callback
  if config.callbacks.get("progressive_resizing", None) is not None:
    num_stages = len(config.callbacks.progressive_resizing.stage_params)
    print(f"Progressive Resizing: {num_stages} stages")
    for i, e in enumerate(config.callbacks.progressive_resizing.stage_params):
      # Stage params are resolution and epochs, pretty print
      print(f"\tStage {i}: {e['resolution']} @ {e['epochs']} epochs")

  # Configure ddp automatically
  n_devices = config.trainer.get('devices', 1)
  if isinstance(n_devices, Sequence):  # trainer.devices could be [1, 3] for example
    n_devices = len(n_devices)
  if n_devices > 1 and config.trainer.get('strategy', None) is None:
    config.trainer.strategy = dict(
      _target_='pytorch_lightning.strategies.DDPStrategy',
      find_unused_parameters=False,
      gradient_as_bucket_view=True,
      # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
    )

  # Init lightning trainer
  log.info(f"Instantiating trainer <{config.trainer._target_}>")
  trainer = hydra.utils.instantiate(
    config.trainer, callbacks=callbacks, logger=logger)

  return trainer


def train(config):
  if config.train.seed is not None:
    pl.seed_everything(config.train.seed, workers=True)

  # Create a trainer
  trainer = create_trainer(config)

  # Create a model
  model = SequenceLightningModule(config)

  # Run initial validation epoch (useful for debugging, finetuning)
  if config.train.validate_at_start:
    print("Running validation before training")
    trainer.validate(model)

  # Fit the model
  if config.train.ckpt is not None:
    trainer.fit(model, ckpt_path=config.train.ckpt)
  else:
    trainer.fit(model)

  # Test the model
  if config.train.test:
    trainer.test(model)


@hydra.main(config_path="../configs", config_name="config.yaml")
def main(config: OmegaConf):
  # Process config:
  # - register evaluation resolver
  # - filter out keys used only for interpolation
  # - optional hooks, including disabling python warnings or debug friendly configuration
  config = utils.train.process_config(config)

  # Pretty print config using Rich library
  utils.train.print_config(config, resolve=True)

  train(config)


if __name__ == "__main__":
  main()