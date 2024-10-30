from torch import nn
import math
import torch
import torch.nn.functional as F
from deep_ssm.mixers.mamba.pscan import pscan


class MambaVsimple(nn.Module):
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
        d_out=None,
        use_cuda = False, # WIll use the mamba selective scan
        pscan: bool = True  #  use parallel scan mode or sequential mode when training
  ):
    """
    Same as deep_ssm.mixers.mamba_simple import Mamba without all the fancy stuff
    """
    super().__init__()
    self.d_model = d_model
    self.d_state = d_state
    self.d_conv = d_conv
    self.expand = expand
    self.d_inner = int(self.expand * self.d_model)
    self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
    self.use_fast_path = use_fast_path
    self.layer_idx = layer_idx

    # additional debugging parameters for very simple model:
    self.use_cuda = use_cuda
    self.pscan = pscan

    #  projects block input from D to 2*ED (two branches)
    self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)

    # Convolutional layer for processing temporal dependencies
    self.conv1d = nn.Conv1d(
        in_channels=self.d_inner,
        out_channels=self.d_inner,
        bias=conv_bias,
        kernel_size=self.d_conv,
        groups=self.d_inner,
        padding=self.d_conv - 1)

    #  projects x to input-dependent delta, B, C
    self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

    #  projects delta from dt_rank to d_inner
    self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

    # Initialize special dt projection to preserve variance at initialization
    dt_init_std = self.dt_rank**-0.5 * dt_scale
    if dt_init == "constant":
      nn.init.constant_(self.dt_proj.weight, dt_init_std)
    elif dt_init == "random":
      nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
    else:
      raise NotImplementedError

    # Bias initialization to control delta (dt) values
    dt = torch.exp(
      torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    inv_dt = dt + torch.log(-torch.expm1(-dt)) #  inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
    with torch.no_grad():
      self.dt_proj.bias.copy_(inv_dt)
    self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit

    # S4D real initialization
    A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
    A_log = torch.log(A)  # Keep A_log in fp32
    self.A_log = nn.Parameter(A_log)
    self.A_log._no_weight_decay = True

    self.D = nn.Parameter(torch.ones(self.d_inner))
    self.D._no_weight_decay = True

    #  projects block output from ED back to D
    if d_out is None:
      d_out = self.d_model
    self.out_proj = nn.Linear(self.d_inner, d_out, bias=bias)

    # CUDA-specific configuration for selective scan
    if self.use_cuda:
      try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        self.selective_scan_cuda = selective_scan_fn
      except ImportError:
        print("Failed to import mamba_ssm. Falling back to mamba.py.")
        self.use_cuda = False

  def forward(self, x):
    #  x : (B, L, D)

    # y : (B, L, D)
    _, L, _ = x.shape

    xz = self.in_proj(x) # (B, L, 2*ED)
    x, z = xz.chunk(2, dim=-1) #  (B, L, ED), (B, L, ED)

    #  x branch
    x = x.transpose(1, 2) #  (B, ED, L)
    x = self.conv1d(x)[:, :, :L] #  depthwise convolution over time, with a short filter
    x = x.transpose(1, 2) #  (B, L, ED)

    x = F.silu(x)

    # Apply state-space model transformation
    y = self.ssm(x, z)

    # Return output directly if using CUDA-accelerated scan
    if self.use_cuda:
      output = self.out_proj(y) # (B, L, D)
      return output # the rest of the operations are done in the ssm function (fused with the CUDA pscan)

    #  z branch
    z = F.silu(z)

    output = y * z
    output = self.out_proj(output) #  (B, L, D)

    return output

  def ssm(self, x, z):
    #  x : (B, L, ED)

    #  y : (B, L, ED)

    A = -torch.exp(self.A_log.float()) # (ED, N)
    D = self.D.float()

    deltaBC = self.x_proj(x) #  (B, L, dt_rank+2*N)
    delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1) #  (B, L, dt_rank), (B, L, N), (B, L, N)
    delta = self.dt_proj.weight @ delta.transpose(1, 2) #  (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
    # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
    # the rest will be applied later (fused if using cuda)

    # choose which selective_scan function to use, according to config
    if self.use_cuda:
      # these are unfortunately needed for the selective_scan_cuda function
      x = x.transpose(1, 2)
      B = B.transpose(1, 2)
      C = C.transpose(1, 2)
      z = z.transpose(1, 2)

      # "softplus" + "bias" + "y * silu(z)" operations are fused
      y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=self.dt_proj.bias.float())
      y = y.transpose(1, 2) # (B, L, ED)

    else:
      delta = delta.transpose(1, 2)
      delta = F.softplus(delta + self.dt_proj.bias)

      if self.pscan:
        y = self.selective_scan(x, delta, A, B, C, D)
      else:
        y = self.selective_scan_seq(x, delta, A, B, C, D)

    return y

  def selective_scan(self, x, delta, A, B, C, D):
    #  x : (B, L, ED)
    #  Δ : (B, L, ED)
    #  A : (ED, N)
    #  B : (B, L, N)
    #  C : (B, L, N)
    #  D : (ED)

    #  y : (B, L, ED)

    deltaA = torch.exp(delta.unsqueeze(-1) * A) #  (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) #  (B, L, ED, N)

    BX = deltaB * (x.unsqueeze(-1)) #  (B, L, ED, N)

    hs = pscan(deltaA, BX)

    y = (hs @ C.unsqueeze(-1)).squeeze(3) #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

    y = y + D * x

    return y

  def selective_scan_seq(self, x, delta, A, B, C, D):
    #  x : (B, L, ED)
    #  Δ : (B, L, ED)
    #  A : (ED, N)
    #  B : (B, L, N)
    #  C : (B, L, N)
    #  D : (ED)

    #  y : (B, L, ED)

    _, L, _ = x.shape

    deltaA = torch.exp(delta.unsqueeze(-1) * A) #  (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) #  (B, L, ED, N)

    BX = deltaB * (x.unsqueeze(-1)) #  (B, L, ED, N)

    h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=deltaA.device) #  (B, ED, N)
    hs = []

    for t in range(0, L):
      h = deltaA[:, t] * h + BX[:, t]
      hs.append(h)

    hs = torch.stack(hs, dim=1) #  (B, L, ED, N)

    y = (hs @ C.unsqueeze(-1)).squeeze(3) #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

    y = y + D * x

    return y
