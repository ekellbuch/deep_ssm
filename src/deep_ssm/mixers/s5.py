import torch
from deep_ssm.mixers.s5_fjax.ssm import S5
import torch.nn as nn
from typing import Optional, List, Tuple
from torch.nn import functional as F



class SequenceLayer(torch.nn.Module):
  def __init__(
    self,
    d_model: int,
    ssm_size: int,
    blocks: int = 1,
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    bidirectional: bool = False,
    C_init: str = "complex_normal",
    conj_sym: bool = False,
    clip_eigs: bool = False,
    step_rescale: float = 1.0,
    discretization: str = "bilinear",
    # layer parameters
    dropout: float = 0.0,
    activation: str = "gelu",
    prenorm: bool = False,
    batchnorm: bool = False,
    bn_momentum: float = 0.9,
    # optional parameters
    bandlimit: float = None,
  ):
    super(SequenceLayer, self).__init__()
    self.d_model = d_model
    self.prenorm = prenorm
    self.batchnorm = batchnorm
    self.activation = activation

    self.seq = S5(
      d_model=d_model,
      ssm_size=ssm_size,
      blocks=blocks,
      dt_min=dt_min,
      dt_max=dt_max,
      bidirectional=bidirectional,
      C_init=C_init,
      conj_sym=conj_sym,
      clip_eigs=clip_eigs,
      step_rescale=step_rescale,
      discretization=discretization,
      bandlimit=bandlimit,
    )

    if self.activation in ["full_glu"]:
      self.out1 = torch.nn.Linear(d_model, d_model)
      self.out2 = torch.nn.Linear(d_model, d_model)
    elif self.activation in ["half_glu1", "half_glu2"]:
      self.out1 = nn.Identity()  # No-op layer
      self.out2 = nn.Linear(d_model, d_model)
    else:
      self.out1 = nn.Identity()
      self.out2 = nn.Identity()

    if self.batchnorm:
      self.norm = torch.nn.BatchNorm1d(d_model, momentum=bn_momentum, track_running_stats=False)
    else:
      self.norm = torch.nn.LayerNorm(d_model)

    self.drop = torch.nn.Dropout(p=dropout)

    self.gelu = F.gelu  # if glu else None

  def apply_activation(self, x):
    # Apply activation
    if self.activation == "full_glu":
      x = self.drop(self.gelu(x))
      out2_result = torch.sigmoid(self.out2(x))
      x = self.out1(x) * out2_result
      x = self.drop(x)
    elif self.activation == "half_glu1":
      x = self.drop(self.gelu(x))
      out2_result = torch.sigmoid(self.out2(x))
      x = x * out2_result
      x = self.drop(x)
    elif self.activation == "half_glu2":
      # Only apply GELU to the gate input
      x1 = self.drop(self.gelu(x))
      out2_result = torch.sigmoid(self.out2(x1))
      x = x * out2_result
      x = self.drop(x)
    elif self.activation == "gelu":
      x = self.drop(self.gelu(x))
    else:
      raise NotImplementedError(
        "Activation: {} not implemented".format(self.activation))
    return x

  def forward(self,
              x: torch.Tensor,
              state: torch.Tensor) -> torch.Tensor:
    """
    """
    skip = x  # (B, L, d_input)
    if self.prenorm:
      if self.batchnorm:
        x = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
      else:
        x = self.norm(x)

    # Apply sequence model
    x, state = self.seq(x, state)  # (B, L, d_input)

    x = self.apply_activation(x)  # (B, L, d_input)

    # residual connection
    x = skip + x

    if not self.prenorm:
      if self.batchnorm:
        x = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
    return x, state  # (B, L, d_input)


class S5Model(nn.Module):

  def __init__(
    self,
    d_input,
    d_output=10,
    ssm_size=384,
    d_model=256,
    n_layers=4,
    blocks: int = 1,
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    bidirectional: bool = False,
    C_init: str = "complex_normal",
    conj_sym: bool = False,
    clip_eigs: bool = False,
    step_rescale: float = 1.0,
    # layer parameters
    dropout: float = 0.0,
    activation: str = "gelu",
    prenorm: bool = False,
    batchnorm: bool = False,
    bn_momentum: float = 0.9,
    bandlimit: float = None,
    discretization: str = "bilinear",

  ):
    super(S5Model, self).__init__()
    self.n_layers = n_layers

    # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
    self.encoder = nn.Linear(d_input, d_model)

    self.layers = nn.Sequential(*[
      SequenceLayer(d_model=d_model,
                    ssm_size=ssm_size,
                    blocks=blocks,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    bidirectional=bidirectional,
                    C_init=C_init,
                    conj_sym=conj_sym,
                    clip_eigs=clip_eigs,
                    step_rescale=step_rescale,
                    discretization=discretization,
                    dropout=dropout,
                    activation=activation,
                    prenorm=prenorm,
                    batchnorm=batchnorm,
                    bn_momentum=bn_momentum,
                    bandlimit=bandlimit,
                    ) for _ in range(self.n_layers)
    ])

    # Linear decoder
    self.decoder = nn.Linear(d_model, d_output)

  def initial_state(self, batch_size):
    # init different A layer:
    states = []
    for layer_idx in range(self.n_layers):
      state_layer_idx = self.layers[layer_idx].seq.initial_state(batch_size)
      states.append(state_layer_idx)
    return states

  def forward(self, x: torch.Tensor, states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor,torch.Tensor]:
    """
    Input x is shape (B, L, d_input)
    """
    output = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

    # Map the layers
    # Each iteration of this loop will map (B, L, d_model) -> (B, L, d_model)
    if states is None:
      states = self.initial_state(x.shape[0])

    # Process each layer with its corresponding state
    new_states = []
    for i, layer in enumerate(self.layers):
        output, state = layer(output, states[i])  # Pass the current state to the current layer
        new_states.append(state)  # Collect the updated state

    # Pooling: average pooling over the sequence length
    output = output.mean(dim=-2)

    # Decode the outputs
    output = self.decoder(output)  # (B, d_model) -> (B, d_output)

    return output, states
