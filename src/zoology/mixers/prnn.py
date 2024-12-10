from deep_ssm.mixers.prnn import pRNN as pRNNBase


class pRNN(pRNNBase):
    def __init__(
        self,
        d_model,
        num_layers,
        layer_idx=None,  # unused
        dropout=0.,
        bias=True,
        batch_first=True,
        bidirectional=False,  # want causal so we don't use bidirectional
        num_iters=2,  # number of iterations for quasi-DEER
        method="minrnn",  # minrrn or gru
        parallel=True,  # parallel implementation
    ):
        super().__init__(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            num_iters=num_iters,
            method=method,
            parallel=parallel,
        )
    
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)[0]  # get output rather than (output, h_n)