import torch
import torch.nn as nn
from deep_ssm.mixers.mamba_extra import MixerModel
from deep_ssm.mixers.prnn import pRNN
import torch.nn.functional as F

class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, model=None):
        super(GRUDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if model is None:
            self.model= nn.GRU(embed_size, hidden_size, 1, batch_first=True)
        else:
            self.model = model
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        if isinstance(self.model, nn.GRU):
            x, h = self.model(x)
        else:
            x = self.model(x)
        x = self.linear(x)
        return x