"""
min_gru.py

Mixers based on minGRU

https://arxiv.org/abs/2410.01201
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


# helper code from Feng '24 for the log space version
def parallel_scan_log(log_coeffs, log_values):
    """
    Args:
      log_coeffs: (B,T,input_size)
      log_values: (B,T+1, input_size)
    """
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)[:, 1:]


def g(x):
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


class minGRUSequenceLayer(nn.Module):
    """
    Sequence layer implementing the minGRU
    """

    def __init__(self, input_size, hidden_size):
        super(minGRUSequenceLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_z = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)

    def forward(self, inputs):
        """
        Args:
          inputs: (B, T, input_size)
        Outputs:
          hiddens: (B, T, hidden_size)
        """
        B, T, _ = inputs.shape
        log_h_0 = log_g(torch.zeros(B, 1, self.hidden_size, device=inputs.device))
        k = self.linear_z(inputs)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_tilde_h = log_g(self.linear_h(inputs))
        hiddens = parallel_scan_log(
            log_coeffs, torch.cat([log_h_0, log_z + log_tilde_h], dim=1)
        )
        return hiddens


class minGRUResidualBlock(nn.Module):
    """
    Residual Block for minGRU
    """

    def __init__(self, input_size, hidden_size):
        super(minGRUResidualBlock, self).__init__()
        self.gru = minGRUSequenceLayer(input_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.RMSNorm(hidden_size)

    def forward(self, x):
        residual = x
        out = self.gru(x)
        out = self.mlp(out)
        out = self.norm(out + residual)
        return out


class minGRUModel(nn.Module):
    """
    Model featuring stacks of minGRU residual blocks
    (i.e. minGRU acting on the sequence, coupled with nonlinearities)
    """

    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super(minGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(minGRUResidualBlock(hidden_size, hidden_size))

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
