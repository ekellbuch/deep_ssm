import torch
import torch.nn as nn
from keras.src.backend.jax.random import dropout

from deep_ssm.mixers.mamba_extra import MixerModel
from deep_ssm.mixers.mamba_simple import Mamba
from deep_ssm.mixers.prnn import pRNN
from deep_ssm.mixers.linear_rnn import RNNModel
from deep_ssm.mixers.lru import LRUModel
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
    def __init__(self, vocab_size,embed_size, hidden_size, num_layers=1, num_iters=2, bidirectional=False,
                 method='gru',
                 parallel=True,
                 k=0.0,
                 clip=False,
                 quasi=True,
                 reparameterize=False,
                 deer_norm=False,
                 dropout=0):
        super(pRNNDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.model= pRNN(embed_size,
                         hidden_size,
                         num_layers=num_layers,
                         batch_first=True,
                         bidirectional=bidirectional,
                         num_iters=num_iters,
                         method=method,
                         parallel=parallel,
                         k=k,
                         clip=clip,
                         quasi=quasi,
                         reparameterize=reparameterize,
                         deer_norm=deer_norm,
                         dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.model(x)
        x = self.linear(x)
        return x

class LinearRNNDecoder(nn.Module):
    """
    implements a deep linear RNN decoder (the linear layers are connected via non-linearities)
    we match the size of the IDS4 matrices from Merrill
    the point is to demonstrate Picard iterations take more layers/iterations than Newton iterations
    currently not training as well as we'd like: should we try HiPPO initialization (in discretized form)?
    Args:
        vocab size: size of the vocabulary
        embed_size: size of the embedding, ends up being d_model in SSM notation
        hidden_size: size of the hidden state, ends up being d_state in SSM notation
    """
    def __init__(
        self,
        vocab_size,
        embed_size,
        hidden_size,
        num_layers=1,
    ):
        super(LinearRNNDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.model = RNNModel(d_input=embed_size, d_output=vocab_size, d_model=embed_size, n_layers=num_layers, d_state=hidden_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.model(x)
        return x

class LRUDecoder(nn.Module):
    """
    implements an LRU decoder
    Args:
        vocab size: size of the vocabulary
        embed_size: size of the embedding, ends up being d_model in SSM notation
        hidden_size: size of the hidden state, ends up being d_state in SSM notation
    """

    def __init__(
        self,
        vocab_size,
        embed_size,
        hidden_size,
        num_layers=1,
    ):
        super(LRUDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.model = LRUModel(
            embed_size,
            vocab_size,
            embed_size,
            num_layers,
            hidden_size=hidden_size,
            r_min=0.0, # see tab 9 of LRU paper for r values used for LRA. We start with list ops as intuitively most similar flavor to state tracking
            r_max=0.99
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.model(x)
        return x


class MambaDecoder(nn.Module):
    def __init__(self, vocab_size,
                 embed_size,
                 d_state=16,
                 d_conv=4,
                 expand_factor=2,
                 num_layers=1,
                 neg_pos_eigenvalues=False,
                 as_complex=False,
                 as_BC_complex_split=False):
        super(MambaDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.model = MixerModel(
            d_model=embed_size,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            n_layer=num_layers,
            mamba_bi_new=True,
            neg_pos_eigenvalues=neg_pos_eigenvalues,
            as_complex=as_complex,
            as_BC_complex_split=as_BC_complex_split,
        )
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.model(x)
        x = self.linear(x)
        return x
