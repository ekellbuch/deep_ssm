
import torch
from torch import nn
from deep_ssm.data.data_transforms import GaussianSmoothing

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


import torch
import torch.nn as nn


class BaseDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        nDays=24,
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        gaussianSmoothSize = 20,
        unfolding=True,
    ):
        """
        The BaseDecoder class is designed to process sequential
        It applies:
        Gaussian smoothing along features
          (neural_dim, gaussianSmoothWidth, gaussianSmoothSize)
        unfolds (extracts extracts sliding windows) along the sequence dimension:
          (kernelLen, strideLen)
        applies a day specific linear transformation and bias to the input
          (nDays)
        """
        super(BaseDecoder, self).__init__()

        self.neural_dim = neural_dim
        self.nDays = nDays
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.gaussianSmoothSize = gaussianSmoothSize
        self.unfolding = unfolding

        # Define the input layer nonlinearity (Softsign activation)
        self.inputLayerNonlinearity = torch.nn.Softsign()

        # Define an unfold operation, which extracts sliding local blocks from a batched input tensor
        # This operation helps in simulating a convolution-like behavior with kernel and stride.
        if self.unfolding:
          self.unfolder = torch.nn.Unfold(
              (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
          )

        # If gaussian smoothing is applied, define a gaussian smoother using the specified width
        # the smoother is applied along the feature dimension
        if self.gaussianSmoothWidth > 0:
            self.gaussianSmoother = GaussianSmoothing(
                self.neural_dim, self.gaussianSmoothSize, self.gaussianSmoothWidth, dim=1
            )

        # Define day-specific weight matrices (learnable parameters) for transforming the input
        # There is one weight matrix per day, with dimensions neural_dim x neural_dim
        self.dayWeights = nn.Parameter(torch.randn(self.nDays, self.neural_dim, self.neural_dim))

        # Define day-specific biases (learnable parameters), one per day, with dimensions 1 x neural_dim
        self.dayBias = nn.Parameter(torch.zeros(self.nDays, 1, self.neural_dim))

        # Initialize dayWeights with identity matrices for each day (ensuring no transformation at the start)
        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(self.neural_dim)

        # Input layers
        #for x in range(nDays):
        #    setattr(self, "inpLayer" + str(x), nn.Linear(self.neural_dim, self.neural_dim))

        #for x in range(nDays):
        #    thisLayer = getattr(self, "inpLayer" + str(x))
        #    thisLayer.weight = nn.Parameter(thisLayer.weight + torch.eye(self.neural_dim))

    def forward_preprocessing(self, neuralInput, dayIdx):
        """

        Args:
          neuralInput: (batch_size x seq_len x num_features)
          dayIdx: (batch_size, )

        Returns:
          stridedInputs: (batch_size x new_seq_len x new_num_features)
        """
        # Smooth along the feature dimension
        if self.gaussianSmoothWidth > 0:
            neuralInput = torch.permute(neuralInput, (0, 2, 1))  # (BS, N, L)
            neuralInput = self.gaussianSmoother(neuralInput)
            neuralInput = torch.permute(neuralInput, (0, 2, 1))  # (BS, L, N)

        # Select the weight matrix for the current day based on dayIdx
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)  # (BS, N, N)

        # Apply a linear transformation to the neural input using the selected day weight matrix
        # This performs a batch-wise matrix multiplication followed by adding the corresponding day bias
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)  # (BS, L, N)

        # Map values between [-1, 1]
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)  # (BS, L, N)

        # Apply the unfold operation extracts sliding windows along seq dimension
        # the feature dimension is expanded by a factor of kernelLen
        # It essentially extracts overlapping blocks of size kernelLen with stride strideLen.
        if self.unfolding:
          stridedInputs = torch.permute(
              self.unfolder(
                  torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)  # (BS, N, L, 1)
              ),
              (0, 2, 1),
          )  # (BS, new L, new N)
        else:
          stridedInputs = transformedNeural

        #assert stridedInputs.shape == (neuralInput.shape[0], (neuralInput.shape[1] - self.kernelLen) // self.strideLen + 1, self.neural_dim*self.kernelLen)
        return stridedInputs


class GRUDecoder(BaseDecoder):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        unfolding=True,
        bidirectional=False,
    ):
        super(GRUDecoder, self).__init__(
            neural_dim=neural_dim,
            nDays=nDays,
            strideLen=strideLen,
            kernelLen=kernelLen,
            gaussianSmoothWidth=gaussianSmoothWidth,
            unfolding=unfolding
        )

        self.layer_dim = layer_dim
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        if unfolding:
          input_dims = self.neural_dim * kernelLen
        else:
          input_dims = self.neural_dim

        self.gru_decoder = nn.GRU(
            input_dims,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        self.fc_decoder_out = nn.Linear(
            self.hidden_dim * (2 if self.bidirectional else 1), n_classes + 1
        )  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        # Preprocess the input (e.g., Gaussian smoothing and unfolding)
        stridedInputs = self.forward_preprocessing(neuralInput, dayIdx)

        # Initialize hidden state
        h0 = torch.zeros(
            self.layer_dim * (2 if self.bidirectional else 1),
            stridedInputs.size(0),
            self.hidden_dim,
            device=neuralInput.device,
        ).requires_grad_()

        # Apply GRU Layer
        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # Apply Decoder
        seq_out = self.fc_decoder_out(hid)
        return seq_out


class MambaDecoder(BaseDecoder):
    def __init__(
        self,
        neural_dim,
        n_classes,
        d_model,
        d_state,
        d_conv,
        expand_factor,
        layer_dim,
        nDays=24,
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional_input=False,
        bidirectional=False,
        unfolding=True,
    ):
        super(MambaDecoder, self).__init__(
            neural_dim=neural_dim,
            nDays=nDays,
            strideLen=strideLen,
            kernelLen=kernelLen,
            gaussianSmoothWidth=gaussianSmoothWidth,
            unfolding=unfolding,
        )
        self.layer_dim = layer_dim
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.bidirectional_input = bidirectional_input

        d_mamba = d_model * 2 if self.bidirectional_input else d_model

        if unfolding:
          input_dims = self.neural_dim * kernelLen
        else:
          input_dims = self.neural_dim


        self.linear_input = nn.Linear(
          input_dims * (2 if self.bidirectional_input else 1), d_mamba
        )

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model=d_mamba,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand_factor,
                    layer_idx=i,
                    bidirectional=bidirectional,
                )
                for i in range(layer_dim)
            ]
        )
        self.norm_f = nn.LayerNorm(d_mamba, eps=1e-5)

        self.fc_decoder_out = nn.Linear(d_mamba, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        stridedInputs = self.forward_preprocessing(neuralInput, dayIdx)

        if self.bidirectional_input:
            stridedFlip = torch.flip(stridedInputs, dims=(1,))
            stridedInputs = torch.cat((stridedInputs, stridedFlip), dim=-1)

        mamba_in = self.linear_input(stridedInputs)

        hidden_states = mamba_in
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        residual = (hidden_states + residual) if residual is not None else hidden_states
        hid = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

        seq_out = self.fc_decoder_out(hid)
        return seq_out



class BidirectionalMamba(nn.Module):
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
    bidirectional=False,
    use_fast_path=False, # True,  # Fused kernel options
    layer_idx=None,
  ):
    # TODO: integrate with model in mixers
    factory_kwargs = {}
    super().__init__()
    self.d_model = d_model
    self.d_state = d_state
    self.d_conv = d_conv
    self.expand = expand
    self.d_inner = int(self.expand * self.d_model)
    self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
    self.use_fast_path = use_fast_path
    self.layer_idx = layer_idx
    self.bidirectional = bidirectional

    if self.bidirectional:
      self.in_proj = nn.Linear(self.d_model, self.d_inner * 3, bias=bias, **factory_kwargs)
    else:
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

    if self.bidirectional:
      self.x_proj = nn.Linear(
        self.d_inner, self.dt_rank * 2 + self.d_state * 4, bias=False, **factory_kwargs
      )
    else:
      self.x_proj = nn.Linear(
        self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
      )

    self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

    # Initialize special dt projection to preserve variance at initialization
    dt_init_std = self.dt_rank ** -0.5 * dt_scale
    if dt_init == "constant":
      nn.init.constant_(self.dt_proj.weight, dt_init_std)
    elif dt_init == "random":
      nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
    else:
      raise NotImplementedError

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

    if self.bidirectional:
      # S4D real initialization
      A_fwd = repeat(
        torch.arange(1, self.d_state + 1, dtype=torch.float32)  ,# , device=device),
        "n -> d n",
        d=self.d_inner,
      ).contiguous()
      A_fwd_log = torch.log(A_fwd)  # Keep A_log in fp32
      self.A_fwd_log = nn.Parameter(A_fwd_log)
      self.A_fwd_log._no_weight_decay = True

      # S4D real initialization
      A_bwd = repeat(
        torch.arange(1, self.d_state + 1, dtype=torch.float32)  ,# , device=device),
        "n -> d n",
        d=self.d_inner,
      ).contiguous()
      A_bwd_log = torch.log(A_bwd)  # Keep A_log in fp32
      self.A_bwd_log = nn.Parameter(A_bwd_log)
      self.A_bwd_log._no_weight_decay = True
    else:
      # S4D real initialization
      A = repeat(
        torch.arange(1, self.d_state + 1, dtype=torch.float32)  ,# , device=device),
        "n -> d n",
        d=self.d_inner,
      ).contiguous()
      A_log = torch.log(A)  # Keep A_log in fp32
      self.A_log = nn.Parameter(A_log)
      self.A_log._no_weight_decay = True

    # D "skip" parameter
    self.D = nn.Parameter(torch.ones(self.d_inner)  )# , device=device))  # Keep in fp32
    self.D._no_weight_decay = True

    if self.bidirectional:
      self.out_proj = nn.Linear(self.d_inner * 2, self.d_model, bias=bias, **factory_kwargs)
    else:
      self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

  def forward(self, hidden_states, inference_params=None):
    """
    hidden_states: (B, L, D)
    Returns: same shape as hidden_states
    """
    batch, seqlen, dim = hidden_states.shape

    conv_state, ssm_state = None, None
    if inference_params is not None:
      conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
      if inference_params.seqlen_offset > 0:
        # The states are updated inplace
        out, _, _ = self.step(hidden_states, conv_state, ssm_state)
        return out

    # We do matmul and transpose BLH -> HBL at the same time
    xz = rearrange(
      self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
      "d (b l) -> b d l",
      l=seqlen,
      )
    if self.in_proj.bias is not None:
      xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    if self.bidirectional:
      A_fwd = -torch.exp(self.A_fwd_log.float())  # (d_inner, d_state)
      A_bwd = -torch.exp(self.A_bwd_log.float())  # (d_inner, d_state)
    else:
      A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

    # In the backward pass we write dx and dz next to each other to avoid torch.cat
    if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
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

      if self.bidirectional:
        x, z = torch.split(xz, [self.d_inner, self.d_inner * 2], dim=1)

        # Compute short convolution
        if conv_state is not None:
          # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
          # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
          conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        if causal_conv1d_fn is None:
          x = self.act(self.conv1d(x)[..., :seqlen])
        else:
          assert self.activation in ["silu", "swish"]
          x = causal_conv1d_fn(
            x=x,
            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation=self.activation,
          )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt_fwd, dt_bwd, B_fwd, B_bwd, C_fwd, C_bwd = torch.split(x_dbl, [self.dt_rank, self.dt_rank,
                                                                         self.d_state, self.d_state,
                                                                         self.d_state, self.d_state], dim=-1)
        dt_fwd = self.dt_proj.weight @ dt_fwd.t()
        dt_fwd = rearrange(dt_fwd, "d (b l) -> b d l", l=seqlen)
        B_fwd = rearrange(B_fwd, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C_fwd = rearrange(C_fwd, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        dt_bwd = self.dt_proj.weight @ dt_bwd.t()
        dt_bwd = rearrange(dt_bwd, "d (b l) -> b d l", l=seqlen)
        B_bwd = rearrange(B_bwd, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C_bwd = rearrange(C_bwd, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        assert self.activation in ["silu", "swish"]
        y_fwd = selective_scan_fn(
          x,
          dt_fwd,
          A_fwd,
          B_fwd,
          C_fwd,
          self.D.float(),
          z=None,
          delta_bias=self.dt_proj.bias.float(),
          delta_softplus=True,
          return_last_state=ssm_state is not None,
        )

        y_bwd = selective_scan_fn(
          torch.flip(x, [-1]),
          dt_bwd,
          A_bwd,
          B_bwd,
          C_bwd,
          self.D.float(),
          z=None,
          delta_bias=self.dt_proj.bias.float(),
          delta_softplus=True,
          return_last_state=ssm_state is not None,
        )

        y = torch.cat((y_fwd, y_bwd), dim=1)
        y *= self.act(z)

        if ssm_state is not None:
          y, last_state = y
          ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

      else:
        x, z = xz.chunk(2, dim=1)

        # Compute short convolution
        if conv_state is not None:
          # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
          # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
          conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        if causal_conv1d_fn is None:
          x = self.act(self.conv1d(x)[..., :seqlen])
        else:
          assert self.activation in ["silu", "swish"]
          x = causal_conv1d_fn(
            x=x,
            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation=self.activation,
          )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
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
        if ssm_state is not None:
          y, last_state = y
          ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
    return out

  def step(self, hidden_states, conv_state, ssm_state):
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
):

  mixer_cls = partial(BidirectionalMamba,
                      layer_idx=layer_idx,
                      d_state=d_state,
                      d_conv=d_conv,
                      expand=expand,
                      bidirectional=bidirectional,
                      )

  norm_cls = partial(
    nn.LayerNorm, eps=norm_epsilon
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
