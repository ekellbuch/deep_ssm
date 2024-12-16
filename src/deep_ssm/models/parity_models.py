import torch
import torch.nn as nn
from deep_ssm.mixers.mamba_extra import MixerModel
from deep_ssm.mixers.prnn import pRNN
import torch.nn.functional as F

class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(GRUDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.model= nn.GRU(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.model(x)
        x = self.linear(x)
        return x
    
class pRNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_iters=2, bidirectional=False,
                 method='minrnn', parallel=True):
        super(pRNNDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.model= pRNN(embed_size, hidden_size, num_layers=1, 
                         batch_first=True, bidirectional=bidirectional, num_iters=num_iters, 
                         method=method, parallel=parallel)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.model(x)
        x = self.linear(x)
        return x