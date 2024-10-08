# Copyright (c) 2023, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except:
    from deep_ssm.mixers.utils_mamba import selective_scan_ref as selective_scan_fn
    from deep_ssm.mixers.utils_mamba import mamba_inner_ref as mamba_inner_fn
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except:
    from deep_ssm.mixers.utils_mamba import causal_conv1d_ref as causal_conv1d_fn
    from deep_ssm.mixers.utils_mamba import causal_conv1d_update_ref as causal_conv1d_update

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except:
    selective_state_update = None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        """
        The Mamba class defines a neural network layer that combines elements of state-space models (SSMs),
        convolutions, and projections to process sequential input data efficiently. The layer is designed
        to handle long-range dependencies in time-series data by leveraging SSMs and performing convolutions
        across local windows of input data. It also supports efficient inference by maintaining and updating
        states across sequences.

        This class is flexible in its configuration, supporting multiple initialization modes, fused kernel
        execution for faster computation, and various options for handling input projections, state updates,
        and inference. It can be used in tasks such as natural language processing, time-series forecasting,
        or any application requiring efficient sequential data processing.

        Parameters:
        -----------
        d_model: int
            The model dimension, i.e., the number of features in the input.

        d_state: int, default=16
            The state expansion factor for the state-space model (SSM). It controls the size of the hidden
            states used to capture long-range dependencies.

        d_conv: int, default=4
            The local convolution width. This controls the kernel size used in the 1D convolution operation
            applied to the input.

        expand: int, default=2
            The block expansion factor that determines the size of intermediate layers within the model.

        dt_rank: str or int, default="auto"
            The rank of the time-step projection matrix. If set to "auto", it is automatically calculated
            based on the model dimension (`d_model`).

        dt_min: float, default=0.001
            Minimum value for the time-step bias initialization.

        dt_max: float, default=0.1
            Maximum value for the time-step bias initialization.

        dt_init: str, default="random"
            Initialization mode for the time-step bias. Can be set to "random" or "constant".

        dt_scale: float, default=1.0
            Scaling factor applied during the initialization of the time-step bias.

        dt_init_floor: float, default=1e-4
            The floor value for the time-step initialization to ensure numerical stability.

        conv_bias: bool, default=True
            Whether to use a bias term in the convolutional layers.

        bias: bool, default=False
            Whether to use a bias term in the linear projection layers.

        use_fast_path: bool, default=True
            Whether to use the fused kernel for faster execution during the forward pass. This enables
            an optimized code path if available.

        layer_idx: int, optional
            Index of the layer, which is useful for caching and managing states during inference. Required
            for certain inference scenarios to maintain state.

        device: torch.device, optional
            The device on which the model's parameters should be allocated (e.g., CPU, GPU).

        dtype: torch.dtype, optional
            The data type of the model's parameters (e.g., torch.float32, torch.float16).
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # projects block input from D to 2*D (two branches)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 5.4: initialize delta_t, the time step, from x_0
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        ### 5: SSM parameter initialization
        # 5.1: Initialize A with evenly spaced eigenvalues;
        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # 5.3: initialize D, the skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        Forward pass of the Mamba layer.

        Args:
        -----
        hidden_states: Tensor of shape (B, L, D)
            - `B`: Batch size.
            - `L`: Sequence length.
            - `D`: Model dimension.

        inference_params: Optional
            - If provided, includes cached states for inference. This allows for faster inference by maintaining
              the previous states from the sequence.

        Returns:
        --------
        Tensor of shape (B, L, D) representing the output of the Mamba layer.
        """
        # Extract the batch size, sequence length, and model dimension from hidden_states
        batch, seqlen, dim = hidden_states.shape

        # Initialize convolutional state and state-space model (SSM) state to None
        conv_state, ssm_state = None, None

        # If inference parameters are provided, retrieve cached states
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)

            # If the sequence length offset is greater than 0, we use a "step" function for inference
            if inference_params.seqlen_offset > 0:
                # Perform a step-by-step update using the cached states and return the output
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # Perform input projection using a linear layer (`in_proj`)
        # The result is rearranged to move the projection step inside the same operation

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )

        # Add bias to the projected input if `in_proj` has bias
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        # Compute the matrix `A` by applying an exponential function to the stored `A_log` parameter
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Fast path for execution: use a fused kernel if available and no inference parameters are provided
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            # Use the fast fused Mamba kernel (`mamba_inner_fn`) for faster computation
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            # Split the input into two parts: `x` (main input) and `z` (modulation signal)
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # Update convolution state using padding and updating inplace:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)

            if causal_conv1d_fn is None:
                # If causal convolution is not available, apply standard convolution with activation (SiLU)
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                # Apply causal convolution with specified activation function (SiLU or Swish)
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # Project the convolved input `x` using `x_proj` and rearrange the result for downstream use
            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)

            # Split the projection result into three components: `dt`, `B`, and `C`
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

            # Apply the `dt_proj` to `dt` and reshape it for further processing
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)

            # Reshape `B` and `C` to their expected shapes
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            # Perform the selective scan operation using the computed parameters
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )

            # Update the state-space model (SSM) state if it's provided
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)

            # Rearrange the output `y` back to shape (B, L, D)
            y = rearrange(y, "b d l -> b l d")

            # Apply the final projection to match the original model dimension
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        """
        Args:
            hidden_states: B, 1, D
            conv_state:  B, 2D, d_conv
            ssm_state: B, 2D, D

        Returns:
            hidden_states: B, 1, D
            conv_state:  B, 2D, d_conv
            ssm_state: B, 2D, D

        """
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """
        Allocates memory for the convolutional and SSM states during inference.
        """
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        """
        Retrieves or initializes the cached states for the layer during inference.
        """
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
