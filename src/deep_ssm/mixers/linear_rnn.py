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
        super(LinearRNNSequenceLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

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
            h = self.sigma(self.A(h) + self.B(u_t))  # Compute next hidden state
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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
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

    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super(LinearRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LinearRNNResidualBlock(hidden_size, hidden_size))

    def forward(self, inputs):
        """
        Args:
          inputs: (B, T, input_size)
        Outputs:
          x: (B, T, hidden_size)
        """
        x = self.input_projection(inputs)
        for layer in self.layers:
            x = layer(x)
        return x, None
