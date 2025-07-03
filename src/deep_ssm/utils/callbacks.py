from lightning import Callback
from lightning.pytorch.utilities import grad_norm
import torch
import torch.nn.functional as F


class GradNormCallback_vars_pbatch(Callback):
    """
    Logs gradnorm in batch
    """
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        #TODO:time grad_norm call vs detach().data.norm(2):
        norms = grad_norm(pl_module.model, norm_type=2)  # You can change `norm_type` to 1 for L1 norm
        for layer_name, norm_value in norms.items():
            pl_module.log(f'{layer_name}', norm_value)


class GradNormCallback_vars_pepoch(Callback):
    """
    Logs gradnorm in epoch
    """
    def on_train_epoch_end(self, trainer, pl_module):
        norms = grad_norm(pl_module.model, norm_type=2)  # You can change `norm_type` to 1 for L1 norm
        for layer_name, norm_value in norms.items():
            pl_module.log(f'{layer_name}', norm_value)


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


class EigenValTracking_pbatch(Callback):
    """
    Logs gradnorm in batch
    """
    def __init__(self):
        self.eigen_has_logged = False

    def on_after_backward(self, trainer, pl_module):
        if self.eigen_has_logged:
            # just do it once
            return
        
        # log eigenvalues:
        named_params = dict(pl_module.model.named_parameters())
        # mamba keywords:
        keywords = ['mamba_fwd.A_log', 'mamba_bwd.A_log', 'x_proj.weight', 'out_proj.weight', 'D']
        # s5 keywords:
        keywords += ['Lambda_re']
        # gru keywords:
        # keywords += ['weight_hh_l0', 'weight_ih_l0']
        for name, param in named_params.items():
            if param.requires_grad:
                if any(keyword in name for keyword in keywords):
                    # let's track val post discretization:
                    if "A_log" in name:
                        # continuous-time A:
                        A = -torch.exp(param)

                        # get learned discretization step dt:
                        dt_weight_name = name.replace("A_log", "dt_proj.weight")
                        dt_bias_name = name.replace("A_log", "dt_proj.bias")
                        dt_weight = named_params[dt_weight_name]  # [d_model, 1]
                        dt_bias = named_params[dt_bias_name]      # [d_model]
                        # Create dummy input to compute dt
                        d_model = A.shape[0]
                        d_state2 = dt_weight.shape[1]
                        dummy_input = torch.ones((1,d_state2), device=param.device, dtype=param.dtype)
                        dt = F.linear(dummy_input, dt_weight, dt_bias).squeeze(-1)
                        dt = F.softplus(dt).squeeze(0)  # ensure positivity

                        # Compute discretized eigenvalues (Λ̄ ≈ exp(Δ * Λ))
                        Lambda = torch.einsum("d,dn->dn", dt, A)  # shape: [d_model, d_state]
                        dA = torch.exp(Lambda)

                        # Check whether the recurrence is stable: discretized eigenvaluye decay
                        pl_module.log(f"stability/max_abs_dA_{name}", dA.abs().max().detach().item())
                        pl_module.log(f"stability/min_abs_dA_{name}", dA.abs().min().detach().item())
                        pl_module.log(f"stability/mean_abs_dA_{name}", dA.abs().mean().detach().item())

                        # Log timescale (memory length) = long time scale = slow decay = longer memory, if too large grad vanish problems
                        tau = -1.0 / (Lambda.real + 1e-8)
                        pl_module.log(f"timescale/mean_tau_{name}", tau.mean().detach().item())
                        # Log oscillation: imaginary part of the eigenvalues, high 
                        if Lambda.dtype.is_complex:
                            pl_module.log(f"oscillation/mean_imag_{name}", Lambda.imag.mean().detach().item())
                        else:
                            pl_module.log(f"oscillation/mean_imag_{name}", 0)

                    elif "x_proj.weight" in name:
                        # if the norm of B and C collapse to 0 recurrence is not used.
                        # if C >> B: the model is not using the input.
                        pl_module.log(f"B/norm_{name}", param.norm().item())
                        pl_module.log(f"B/max_{name}", param.abs().max().item())
                    elif "out_proj.weight" in name:
                        pl_module.log(f"C/norm_{name}", param.norm().item())
                        pl_module.log(f"C/max_{name}", param.abs().max().item())
                    elif name.endswith("D"):
                        pl_module.log(f"D/norm_{name}", param.norm().item())
                        pl_module.log(f"D/max_{name}", param.abs().max().item())
                    # S5 model
                    elif "Lambda_re" in name:
                        layer = name.split('.',3)[2]
                        layer_idx = int(layer)
                        prefix = f"backbone.layers.{layer}.seq.seq"
                        layer_name = name.replace("backbone.layers.", "").replace(".seq.seq", "") 
                        Lambda_re = named_params[f"{prefix}.Lambda_re"]
                        Lambda_im = named_params[f"{prefix}.Lambda_im"]
                        log_step  = named_params[f"{prefix}.log_step"]
                        B = named_params[f"{prefix}.B"]
                        C = named_params[f"{prefix}.C"]
                        D = named_params[f"{prefix}.D"]

                        C_tilde = torch.complex(C[..., 0], C[..., 1])
                        B_tilde = torch.complex(B[..., 0], B[..., 1])

                        step_scale = torch.exp(log_step[..., 0])
                        Lambda_bar, B_bar =  pl_module.model.backbone.layers[layer_idx].seq.seq.discretize(Lambda_re, B_tilde, step_scale)

                        # We wto check statbility
                        self.log(f"layer{layer_name}/Lambda_bar_re_max", Lambda_bar.abs().max().item())
                        self.log(f"layer{layer_name}/Lambda_bar_re_min", Lambda_bar.abs().min().item())
                        self.log(f"layer{layer_name}/Lambda_bar_re_mean", Lambda_bar.abs().mean().item())

                        # Log oscillation: Im(Λ)
                        pl_module.log(f"layer{layer_name}/Lambda_bar_im_mean", Lambda_im.mean().item())
                        pl_module.log(f"layer{layer_name}/Lambda_bar_im_max", Lambda_im.max().item())
                        pl_module.log(f"layer{layer_name}/Lambda_bar_im_min", Lambda_im.min().item())

                        # Log step scale (Δ)
                        pl_module.log(f"layer{layer_name}/step_mean", step_scale.mean().item())
                        pl_module.log(f"layer{layer_name}/step_max", step_scale.max().item())
                        pl_module.log(f"layer{layer_name}/step_min", step_scale.min().item())

                        # Optional: log norms of B, C, D
                        pl_module.log(f"layer{layer_name}/B_norm", B_tilde.abs().norm().item())
                        pl_module.log(f"layer{layer_name}/C_norm", C_tilde.abs().norm().item())
                        pl_module.log(f"layer{layer_name}/D_norm", D.norm().item())
                    else:
                        pass
        norms = grad_norm(pl_module.model, norm_type=2)
        for layer_name, norm_value in norms.items():
            pl_module.log(f'{layer_name}', norm_value)

        self.eigen_has_logged = True

    def on_train_epoch_end(self, trainer, pl_module):
        self.eigen_has_logged = False


class CERCallback(Callback):
    def __init__(self):
        self.cer_outputs = {
            "train": 0,
            "validation": 0,
            "test": 0,
        }
        self.cer_counts = {
            "train": 0,
            "validation": 0,
            "test": 0,
        }

    def _compute_and_store(self, pl_module, batch, flag):
        X, y, X_len, y_len, dayIdx = batch
        input_len = pl_module.get_lens(X_len)
        preds = pl_module.model(X, dayIdx)
        cer_v = pl_module.calculate_cer(preds, input_len, y, y_len)
        self.cer_outputs[flag] += cer_v
        self.cer_counts[flag] += 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._compute_and_store(pl_module, batch, flag="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._compute_and_store(pl_module, batch, flag="validation")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._compute_and_store(pl_module, batch, flag="test")

    def _finalize_epoch(self, pl_module, flag):
        if self.cer_counts[flag] > 0:
            epoch_cer = self.cer_outputs[flag] / self.cer_counts[flag]
            pl_module.log(f"cer_{flag}_epoch", epoch_cer)
            self.cer_outputs[flag] = 0
            self.cer_counts[flag] = 0

    def on_train_epoch_end(self, trainer, pl_module):
        self._finalize_epoch(pl_module, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._finalize_epoch(pl_module, "validation")

    def on_test_epoch_end(self, trainer, pl_module):
        self._finalize_epoch(pl_module, "test")


all_callbacks = {
    "model_pbatch": GradNormCallback(),
    "model_pepoch": GradNormCallback_pepoch(),
    "vars_pbatch": GradNormCallback_vars_pbatch(),
    "vars_pepoch": GradNormCallback_vars_pepoch(),
    "masking_scheduler": UpdateMaskingStrategy,
    "eigen_track": EigenValTracking_pbatch(),
    "cer" : CERCallback(),
}