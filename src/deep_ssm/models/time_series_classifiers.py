"""
time_series_classifiers.py
Build some models to do time series classification, building off of prnn
"""
import torch
import torch.nn as nn

import pdb

from deep_ssm.mixers.prnn import pRNN
from deep_ssm.mixers.min_gru import minGRUModel


class pRNNClassifier(nn.Module):
    """
    Classifier built on pRNN for time series classification

    Note: follows batch first convention
    Note: follows input_size, hidden_size pytorch convention (https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
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
                 family="prnn", # alternatively, "min_gru"
                 use_mean=False, # should the summary statistic be mean or last hidden state?
                 num_classes=10):
        super(pRNNClassifier, self).__init__()
        if family=="prnn":
            self.rnn = pRNN(input_size,
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
                            while_loop=while_loop)
        elif family=="min_gru":
            self.rnn = minGRUModel(input_size, hidden_size, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown family: {family}")
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.use_mean = use_mean

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        returns: (batch_size, num_classes)
        """
        # pdb.set_trace()
        rnn_output, _ = self.rnn(x)

        if self.use_mean:
            sum_stat = torch.mean(rnn_output, dim=1)
        else:
            sum_stat = rnn_output[:, -1, :]

        # Pass through the classifier
        logits = self.classifier(sum_stat)  # (batch_size, num_classes)

        return logits

    def predict(self, x):
        """
        Convenience method for getting class predictions
        """
        logits = self(x)
        return torch.argmax(logits, dim=1)
