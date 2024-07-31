import os
import random
import wandb
import time
from pytorch_lightning.loggers import WandbLogger
from typing import Callable, List, Sequence
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from functools import partial, wraps


class DummyExperiment:
  """Dummy experiment."""

  def nop(self, *args, **kw):
    pass

  def __getattr__(self, _):
    return self.nop

  def __getitem__(self, idx) -> "DummyExperiment":
    # enables self.logger.experiment[0].add_image(...)
    return self

  def __setitem__(self, *args, **kwargs) -> None:
    pass


def rank_zero_experiment(fn: Callable) -> Callable:
  """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

  @wraps(fn)
  def experiment(self):
    @rank_zero_only
    def get_experiment():
      return fn(self)

    return get_experiment() or DummyExperiment()

  return experiment


class CustomWandbLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        """Modified logger that insists on a wandb.init() call and catches wandb's error if thrown."""

        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
        .. code-block:: python
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        print("wandb Exception:\n", e)
                        t = random.randint(30, 60)
                        print(f"Sleeping for {t} seconds")
                        time.sleep(t)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment