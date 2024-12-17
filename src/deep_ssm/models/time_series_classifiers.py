"""
time_series_classifiers.py
Build some models to do time series classification, building off of prnn
"""
import torch
import torch.nn as nn

from deep_ssm.mixers.prnn import pRNN


class pRNNClassifier(nn.Module):
    """
    Classifier built on MinRNN for time series classification

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
                 method="minrnn",
                 parallel=True,
                 checkpoint=True,
                 learn_diags=False,
                 num_classes=10):
        super(pRNNClassifier, self).__init__()
        self.rnn = pRNN(input_size,
                        hidden_size,
                        num_layers=num_layers,
                        batch_first=batch_first,
                        bidirectional=bidirectional,
                        num_iters=num_iters,
                        k=k,
                        method=method,
                        parallel=parallel,
                        checkpoint=checkpoint,
                        learn_diags=learn_diags)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        returns: (batch_size, num_classes)
        """
        rnn_output, _ = self.rnn(x)

        # Use the last output for classification
        last_hidden = rnn_output[:, -1, :]  # (batch_size, hidden_size)

        # Pass through the classifier
        logits = self.classifier(last_hidden)  # (batch_size, num_classes)

        return logits

    def predict(self, x):
        """
        Convenience method for getting class predictions
        """
        logits = self(x)
        return torch.argmax(logits, dim=1)
