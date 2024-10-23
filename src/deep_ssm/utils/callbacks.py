from lightning import Callback
from lightning.pytorch.utilities import grad_norm


class GradNormCallback_vars_pbatch(Callback):
    """
    Logs gradnorm in batch
    """
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        #TODO:time grad_norm call vs detach().data.norm(2):
        norms = grad_norm(pl_module.model, norm_type=2)  # You can change `norm_type` to 1 for L1 norm
        for layer_name, norm_value in norms.items():
            self.log(f'{layer_name}', norm_value)


class GradNormCallback_vars_pepoch(Callback):
    """
    Logs gradnorm in epoch
    """
    def on_train_epoch_end(self, trainer, pl_module):
        norms = grad_norm(pl_module.model, norm_type=2)  # You can change `norm_type` to 1 for L1 norm
        for layer_name, norm_value in norms.items():
            self.log(f'{layer_name}', norm_value)


class GradNormCallback(Callback):
    """
    Logs gradnorm at the end of the backward pass
  Edited from https://github.com/Lightning-AI/lightning/issues/1462
  """
    def on_after_backward(self, trainer, pl_module):
        pl_module.log("grad_norm/model", gradient_norm(pl_module))


class GradNormCallback_pepoch(Callback):
    """
    Logs gradnorm at the end of the epoch
    Edited from https://github.com/Lightning-AI/lightning/issues/1462
    """

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log("grad_norm/model", gradient_norm(pl_module))


def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item()**2
    total_norm = total_norm**(1. / 2)
    return total_norm


class UpdateMaskingStrategy(Callback):

    def __init__(self,
                 masking_epochs=[2, 3],
                 speckled_mask_p_schedule=[0.1, 0.2],
                 temporal_mask_n_schedule=[0, 1],
                 feature_mask_p_schedule=[0, 0]
                 ):
        """
        Args:
            args: Configuration arguments for dataloaders and other training parameters.
            masking_epochs: List of epochs at which masking strategies should change.
            speckled_mask_p_schedule: List of speckled masking probabilities to apply at each epoch.
            temporal_mask_n_schedule: List of temporal masking steps to apply at each epoch.
            dataloader_epoch_interval: Interval (in epochs) to reload the dataloaders.
        """
        self.masking_epochs = masking_epochs
        self.speckled_mask_p_schedule = speckled_mask_p_schedule
        self.temporal_mask_n_schedule = temporal_mask_n_schedule
        self.feature_mask_p_schedule = feature_mask_p_schedule

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # Update dataloader configuration dynamically
        if epoch in self.masking_epochs:
            # Dynamically update masking parameters
            index = self.masking_epochs.index(epoch)
            speckled_mask_p = self.speckled_mask_p_schedule[index]
            temporal_mask_n = self.temporal_mask_n_schedule[index]
            feature_mask_p = self.feature_mask_p_schedule[index]

            # Update args with new masking values
            trainer.datamodule.args.speckled_mask_p = speckled_mask_p
            trainer.datamodule.args.temporal_mask_n = temporal_mask_n
            trainer.datamodule.args.feature_mask_p = feature_mask_p

            # Update train transform
            trainer.datamodule.update_transforms()


all_callbacks = {
    "model_pbatch": GradNormCallback(),
    "model_pepoch": GradNormCallback_pepoch(),
    "vars_pbatch": GradNormCallback_vars_pbatch(),
    "vars_pepoch": GradNormCallback_vars_pepoch(),
    "masking_scheduler": UpdateMaskingStrategy,
}