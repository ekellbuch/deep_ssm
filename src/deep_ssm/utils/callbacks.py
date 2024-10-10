from lightning import Callback
from lightning.pytorch.utilities import grad_norm


class GradNormCallback_vars_pbatch(Callback):
    """
    Logs the gradient norm for each parameter individually.
    """
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        #TODO:time grad_norm call vs detach().data.norm(2):
        norms = grad_norm(pl_module.model, norm_type=2)  # You can change `norm_type` to 1 for L1 norm
        for layer_name, norm_value in norms.items():
            self.log(f'grad_norm_{layer_name}', norm_value, on_step=True, on_epoch=False)


class GradNormCallback_vars_pepoch(Callback):
    """
    Logs the gradient norm for each parameter individually.
    """

    def on_train_epoch_end(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                # Calculate the norm of the gradient
                param_grad_norm = param.grad.detach().data.norm(2).item()
                # Log the gradient norm for this parameter
                pl_module.log(f"grad_norm/{name}", param_grad_norm)

        # Optionally, log the total gradient norm
        total_grad_norm = gradient_norm(pl_module)
        pl_module.log("grad_norm/model", total_grad_norm)



class GradNormCallback(Callback):
    """
  Logs the gradient norm.
  Edited from https://github.com/Lightning-AI/lightning/issues/1462
  """

    def on_after_backward(self, trainer, pl_module):
        pl_module.log("grad_norm/model", gradient_norm(pl_module))


class GradNormCallback_pepoch(Callback):
    """
    Logs the gradient norm.
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