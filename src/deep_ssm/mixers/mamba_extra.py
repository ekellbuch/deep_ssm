import torch
from torch import nn

from functools import partial
import math
import torch.nn.functional as F
from einops import rearrange, repeat

try:
  from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except:
  from deep_ssm.mixers.utils_mamba import causal_conv1d_ref as causal_conv1d_fn
  from deep_ssm.mixers.utils_mamba import causal_conv1d_update_ref as causal_conv1d_update

try:
  from mamba_ssm.modules.block import Block
  from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
  from mamba_ssm.ops.triton.selective_state_update import selective_state_update
  from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except:
  from deep_ssm.mixers.utils_mamba import selective_scan_ref as selective_scan_fn
  from deep_ssm.mixers.utils_mamba import Block_ref as Block
  from deep_ssm.mixers.utils_mamba import RMSNorm_ref as RMSNorm
  from deep_ssm.mixers.utils_mamba import layer_norm_ref as layer_norm_fn
  from deep_ssm.mixers.utils_mamba import rms_norm_ref as rms_norm_fn
  selective_state_update = None

from typing import Optional
from deep_ssm.mixers.mamba_simple import Mamba


def create_block(
  d_model, # Model dimension d_model
  d_state,  # SSM state expansion factor
  d_conv,    # Local convolution width
  expand,    # Block expansion factor
  norm_epsilon=1e-5,
  rms_norm=False,
  residual_in_fp32=False,
  fused_add_norm=False,
  layer_idx=None,
  bidirectional=False,
  mamba_bi_new=False,
  bidirectional_strategy=None,
):
  if mamba_bi_new:
    model = MambaWrapper
  else:
    raise NotImplementedError("Only MambaWrapper is supported for now.")


  mixer_cls = partial(model,
                      layer_idx=layer_idx,
                      d_state=d_state,
                      d_conv=d_conv,
                      expand=expand,
                      bidirectional=bidirectional,
                      bidirectional_strategy=bidirectional_strategy,
                      )

  norm_cls = partial(
    nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon
  )
  # TODO: check if mlp_cls was in previous version
  mlp_cls = nn.Identity

  block = Block(
    d_model,
    mixer_cls,
    mlp_cls,
    norm_cls=norm_cls,
    fused_add_norm=fused_add_norm,
    residual_in_fp32=residual_in_fp32,
  )
  block.layer_idx = layer_idx
  return block


class MambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality."
        TODO: compare with bidirectional in audio model
    """
    def __init__(
        self,
        d_model: int,
        bidirectional: bool = False,
        bidirectional_strategy: Optional[str] = None,
        **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"  # Default strategy: `add`
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!")
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(
            d_model=d_model,
            **mamba_kwargs
        )
        if bidirectional:
            self.mamba_rev = Mamba(
                d_model=d_model,
                **mamba_kwargs
            )
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, inference_params=None):
        """Bidirectional-enabled forward pass

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(dims=(1,)),  # Flip along the sequence length dimension
                inference_params=inference_params
            ).flip(dims=(1,))  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
        return out


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        d_conv,
        expand_factor,
        n_layer,
        norm_epsilon=1e-5,
        rms_norm=False,
        fused_add_norm=False,
        residual_in_fp32=False,
        bidirectional=False,
        mamba_bi_new=True,
        initialize_mixer=False,
        bidirectional_strategy=None,
    ):
        super().__init__()

        self.residual_in_fp32 = residual_in_fp32
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.fused_add_norm = fused_add_norm

        # Block of model layers
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand_factor,
                    layer_idx=i,
                    norm_epsilon=norm_epsilon,
                    bidirectional=bidirectional,
                    mamba_bi_new=mamba_bi_new,
                    fused_add_norm=fused_add_norm,
                    rms_norm=rms_norm,
                    bidirectional_strategy=bidirectional_strategy,
                )
                for i in range(n_layer)
            ]
        )
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon)

        if initialize_mixer:
          self.apply(
              partial(
                  _init_weights,
                  n_layer=n_layer,
                  n_residuals_per_layer=1,  # 2 if we have MLP
              )
        )

    def forward(self, hidden_states, inference_params=None, **mixer_kwargs):

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)#, inference_params=inference_params, **mixer_kwargs)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
              hidden_states,
              self.norm_f.weight,
              self.norm_f.bias,
              eps=self.norm_f.eps,
              residual=residual,
              prenorm=False,
              residual_in_fp32=self.residual_in_fp32,
              is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states