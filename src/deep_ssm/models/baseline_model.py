import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from deep_ssm.mixers.s5 import SequenceLayer
from deep_ssm.mixers.s4 import S4Block


class S5Model(nn.Module):
    """S5 model for sequential tasks like MNIST/CIFAR."""
    
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
            SequenceLayer(
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
        """Initialize states for all layers."""
        states = []
        for layer_idx in range(self.n_layers):
            state_layer_idx = self.layers[layer_idx].seq.initial_state(batch_size)
            states.append(state_layer_idx)
        return states

    def forward(self, x: torch.Tensor, states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through S5 model.
        Input x is shape (B, L, d_input)
        """
        output = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        # Initialize states if not provided
        if states is None:
            states = self.initial_state(x.shape[0])

        # Process each layer with its corresponding state
        new_states = []
        for i, layer in enumerate(self.layers):
            output, state = layer(output, states[i])
            new_states.append(state)

        # Pooling: average pooling over the sequence length
        output = output.mean(dim=-2)

        # Decode the outputs
        output = self.decoder(output)  # (B, d_model) -> (B, d_output)

        return output, new_states


class S4Model(nn.Module):
    """S4 model for sequential tasks like MNIST/CIFAR."""
    
    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout: float = 0.0,
        prenorm: bool = False,
        transposed: bool = True,
        lr: float = 0.001,
        **layer_args,
    ):
        super(S4Model, self).__init__()
        self.n_layers = n_layers
        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4Block(d_model, dropout=dropout, transposed=transposed, lr=lr,**layer_args)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout1d(dropout) if dropout > 0 else nn.Identity())

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor, states=None) -> Tuple[torch.Tensor, None]:
        """
        Forward pass through S4 model.
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            
            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x, None  # Return None for states to match S5Model interface
