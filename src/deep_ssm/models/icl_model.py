"""
time_series_classifiers.py
Build some models to do time series classification, building off of prnn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from deep_ssm.mixers.prnn import pRNN
from deep_ssm.mixers.min_gru import minGRUModel


class GLU(nn.Module):
    # Input: (B, L, D_in)
    # Output: (B, L, D_out)
    def __init__(self, in_features, out_features):
        super(GLU, self).__init__()
        self.out1 = nn.Linear(in_features, out_features)
        self.out2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.out1(x) * torch.sigmoid(self.out2(x))


class pRNNRegressor(nn.Module):
    """
    Classifier built on pRNN for time series classification

    Note: follows batch first convention
    Note: follows input_size, hidden_size pytorch convention (https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        output_size=1,
        batch_first=True,
        bidirectional=False,
        num_iters=10,
        k=0.0,
        clip=False,
        method="gru",
        parallel=True,
        checkpoint=True,
        learn_diags=False,
        last_only=False,
        learn_initial_states=False,
        while_loop=False,
        quasi=True,
        family="prnn",  # alternatively, "min_gru"
        dropout=0.0,
    ):
        super(pRNNRegressor, self).__init__()
        self.input_layer = nn.Linear(in_features=input_size, out_features=hidden_size)
        if family == "prnn":
            # Add MLPs in between each RNN layer
            self.rnns = nn.ModuleList(
                [
                    pRNN(
                        hidden_size,
                        hidden_size,
                        num_layers=num_layers,
                        batch_first=batch_first,
                        bidirectional=bidirectional,
                        num_iters=num_iters,
                        k=k,
                        clip=clip,
                        method=method,
                        parallel=parallel,
                        checkpoint=checkpoint,
                        learn_diags=learn_diags,
                        last_only=last_only,
                        learn_initial_states=learn_initial_states,
                        while_loop=while_loop,
                        quasi=quasi,
                        dropout=dropout,
                    ),
                ]
            )
            self.glus = nn.ModuleList(
                [GLU(hidden_size, hidden_size) for _ in range(num_layers - 1)]
            )
            self.lns = nn.ModuleList(
                [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
            )
        elif family == "min_gru":
            self.rnn = minGRUModel(input_size, hidden_size, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown family: {family}")
        if bidirectional:
            hidden_size = hidden_size * 2
        self.classifier = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        returns: (batch_size, num_classes)
        """
        x = self.input_layer(x)
        for i in range(len(self.rnns)):
            x, _ = self.rnns[i](x)
            # Apply GELU activation
            x = F.gelu(x)
            x = self.dropout(x)
            # Apply GLU
            if i < len(self.rnns) - 1:
                x = self.glus[i](x)
            x = self.lns[i](x)
        logits = self.classifier(x)
        return logits
