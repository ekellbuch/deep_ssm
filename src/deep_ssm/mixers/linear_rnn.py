"""
Code for a deep linear RNN sequence model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRNNSequenceLayer(nn.Module):
    """
    Sequence layer implementing a dense linear RNN
    Currently evaluated sequentially
    """

    def __init__(self, input_size, hidden_size):
        """
        Args:
            input_size: aka d_model
            hidden_size: aka d_state
        """
        super(LinearRNNSequenceLayer, self).__init__()
        self.input_size = input_size # aka d_model, 255 for IDS4
        self.hidden_size = hidden_size # aka d_state, 110 for IDS4

        # Weight matrices
        self.A = nn.Linear(hidden_size, hidden_size, bias=False)  # hidden-to-hidden
        self.B = nn.Linear(input_size, hidden_size, bias=False)  # input-to-hidden
        self.C = nn.Linear(
            hidden_size, self.input_size, bias=False
        )  # hidden-to-output
        self.D = nn.Linear(input_size, self.input_size, bias=False)  # input-to-output

        # Initialize weights orthogonally
        nn.init.orthogonal_(self.A.weight)
        nn.init.orthogonal_(self.B.weight)
        nn.init.orthogonal_(self.C.weight)
        nn.init.orthogonal_(self.D.weight)

    def forward(self, inputs):
        """
        Args:
          inputs: (B, T, input_size)
        Outputs:
          outputs: (B, T, input_size)
        """
        B, T, _ = inputs.shape
        h_init = torch.zeros(B, self.hidden_size, device=inputs.device)
        # Iterate through the sequence
        outputs = []
        h = h_init
        for t in range(T):
            u_t = inputs[:, t, :]  # Get input for time step t
            h = self.A(h) + self.B(u_t) # Compute next hidden state
            y = self.C(h) + self.D(u_t)  # Compute output
            outputs.append(y.unsqueeze(1))  # Append to outputs list
        outputs = torch.cat(outputs, dim=1)
        return outputs


class LinearRNNResidualBlock(nn.Module):
    """
    Residual Block for minGRU
    """

    def __init__(self, input_size, hidden_size):
        super(LinearRNNResidualBlock, self).__init__()
        self.sequence_mixer = LinearRNNSequenceLayer(input_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.norm = nn.RMSNorm(hidden_size)

    def forward(self, x):
        residual = x
        out = self.sequence_mixer(x)
        out = self.mlp(out)
        out = self.norm(out + residual)
        return out


class LinearRNNModel(nn.Module):
    """
    Model featuring stacks of LinearRNN residual blocks
    (i.e. LinearRNN acting on the sequence, coupled with nonlinearities)
    """

    def __init__(self, d_model=255, d_state=110, num_layers=1, input_size=None):
        """
        Args:
            d_model: in SSM notation, the dimension of the inputs us and the outputs ys
            d_state: in SSM notation, the dimension of the hidden states xs
            num_layers: the number of residual blocks
            input_size: if None, assumes same as d_model
        """
        super(LinearRNNModel, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        self.input_size = input_size
        if self.input_size is not None:
            self.input_projection = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LinearRNNResidualBlock(self.d_model, self.d_state))

    def forward(self, inputs):
        """
        Args:
          inputs: (B, T, input_size)
        Outputs:
          x: (B, T, d_model)
        """
        if self.input_size is None:
            assert len(inputs.shape) == 3, f"inputs shape: {inputs.shape}, but it needs to be (B,T,d_model)"
            assert inputs.shape[-1] == self.d_model, f"inputs have d_model {inputs.shape[-1]}, but it needs to be {self.d_model}"
            x = inputs
        else:
            x = self.input_projection(inputs)
        for layer in self.layers:
            x = layer(x)
        return x, None
