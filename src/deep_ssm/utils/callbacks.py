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


all_callbacks = {
    "model_pbatch": GradNormCallback(),
    "model_pepoch": GradNormCallback_pepoch(),
    "vars_pbatch": GradNormCallback_vars_pbatch(),
    "vars_pepoch": GradNormCallback_vars_pepoch(),
}