import torch
import torch.nn as nn
from deep_ssm.mixers.mamba_extra import MixerModel
from deep_ssm.mixers.prnn import pRNN
import torch.nn.functional as F


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(GRUDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.model= nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.model(x)
        x = self.linear(x)
        return x


class pRNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, num_iters=2, bidirectional=False,
                 method='gru', parallel=True, k=0.0, clip=False, quasi=True, reparameterize=True):
        super(pRNNDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.model= pRNN(embed_size, hidden_size, num_layers=num_layers, 
                         batch_first=True, bidirectional=bidirectional, num_iters=num_iters, 
                         method=method, parallel=parallel, k=k, clip=clip, quasi=quasi, reparameterize=reparameterize)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.model(x)
        print(x)
        x = self.linear(x)
        print(x)
        return x


class MambaDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, d_state=16, d_conv=4, expand_factor=2, num_layers=1, neg_pos_eigenvalues=False):
        super(MambaDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.model = MixerModel(
            d_model=embed_size,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            n_layer=num_layers,
            mamba_bi_new=True,
            neg_pos_eigenvalues=neg_pos_eigenvalues
        )
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.model(x)
        x = self.linear(x)
        return x
