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


class CausalConv1d(nn.Module):
    """
    Adapted from xLSTM: https://github.com/NX-AI/xlstm/blob/5feeeaf11b973e37b960553bd8749ca9dadbdfe4/xlstm/components/conv.py
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,D), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,D)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    def __init__(self, D, kernel_size):
        super().__init__()
        self.groups = D
        self.kernel_size = kernel_size
        self.pad = (
            self.kernel_size - 1
        )  # padding of this size assures temporal causality.
        self.conv = nn.Conv1d(
            in_channels=D,
            out_channels=D,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=self.groups,
        )
        # B, C, L

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B,T,D)
        Returns:
            Tensor of shape (B,T,D)
        """
        if self.kernel_size == 0:
            return x
        y = x.transpose(2, 1)  # (B,D,T) tensor - now in the right shape for conv layer.
        y = self.conv(y)  # (B,D,T+pad) tensor
        return y[:, :, : -self.pad].transpose(2, 1)


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
    Residual Block for minGRU with a causal convolution layer.
    The residual block maps between sequences with the same dimensions
    """

    def __init__(self, D, conv=False):
        """
        D: dimension of the data
        conv: bool, whether to use a convolutional layer or not
        """
        super(minGRUResidualBlock, self).__init__()
        # Causal convolution: kernel_size=4
        self.conv = conv
        if conv:
            self.causal_conv = CausalConv1d(D, 4)
        self.gru = minGRUSequenceLayer(D, D)
        self.mlp = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.ReLU(),
            nn.Linear(4 * D, D),
        )
        self.norm = nn.RMSNorm(D)

    def forward(self, x):
        residual = x

        # Transpose for Conv1d: (batch_size, seq_len, input_size) -> (batch_size, input_size, seq_len)
        if self.conv:
            x = self.causal_conv(x)
        # print(x.shape)
        # Transpose back to (batch_size, seq_len, input_size)

        out = self.gru(x)
        out = self.mlp(out)
        out = self.norm(out + residual)
        return out


class minGRUModel(nn.Module):
    """
    Model featuring stacks of minGRU residual blocks
    (i.e. minGRU acting on the sequence, coupled with nonlinearities)
    """

    def __init__(self, input_size=1, hidden_size=16, num_layers=1, conv=False):
        """
        Args:
            conv: bool, whether to use a convolutional layer or not
        """
        super(minGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList()
        self.conv = conv
        for _ in range(num_layers):
            self.layers.append(minGRUResidualBlock(hidden_size, conv=conv))

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
