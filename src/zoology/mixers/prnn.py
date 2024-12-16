# %%
from deep_ssm.mixers.prnn import pRNN as pRNNBase

import torch
import math

from torch import nn
from einops import rearrange


class FastDepthwiseCausalConv1d(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, bias: bool=True, channel_last=False, activation=None, device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(in_channels, kernel_size, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(in_channels, device=device, dtype=dtype)) if bias else None
        self.activation = activation
        self.channel_last = channel_last
        self.activations_dict = {"silu": torch.nn.functional.silu, None: lambda x: x, "exp": torch.exp}

        self._init_weights()
    
    def _init_weights(self):
        # handle xavier init for depthwise conv, since pytorch is wrong for grouped convs
        # see e.g. https://github.com/pytorch/pytorch/issues/23854
        # or https://github.com/huggingface/pytorch-image-models/issues/84
        fan_in = self.kernel_size
        fan_out = self.kernel_size
        a = math.sqrt(6 / (fan_in + fan_out))
        self.weight.data.uniform_(-a, a)
        nn.init.zeros_(self.bias.data) if self.bias is not None else None

    def forward(self, x):
        if self.channel_last:
            x = x.transpose(-1, -2)
        *batch_dims, dim, seq_len = x.shape
        x = rearrange(x, '... d l -> (...) d l')  # collapse batch dims
        y = self.activations_dict[self.activation](torch.nn.functional.conv1d(x, self.weight[:, None, :], bias=self.bias, groups=self.in_channels, padding=self.kernel_size-1)[..., :seq_len])
        y = y.reshape(*batch_dims, dim, seq_len)  # expand batch dims
        if self.channel_last:
            y = y.transpose(-1, -2)
        return y


class pRNN(pRNNBase):
    def __init__(
        self,
        d_model,
        d_conv,
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
        self.conv1d = FastDepthwiseCausalConv1d(d_model, kernel_size=d_conv, bias=True, channel_last=True)
    
    def forward(self, x):
        x = self.conv1d(x)
        return super().forward(x)[0]  # get output rather than (output, h_n)
# %%
#import gc
#
#def clear_device_cache(garbage_collection=False):
#    """
#    Clears the device cache by calling `torch.{backend}.empty_cache`. Can also run `gc.collect()`, but do note that
#    this is a *considerable* slowdown and should be used sparingly.
#    """
#    if garbage_collection:
#        gc.collect()
#    torch.cuda.empty_cache()
#
#
#def release_memory(*objects):
#    """
#    Releases memory from `objects` by setting them to `None` and calls `gc.collect()` and `torch.cuda.empty_cache()`.
#    Returned objects should be reassigned to the same variables.
#    Args:
#        objects (`Iterable`):
#            An iterable of objects
#    Returns:
#        A list of `None` objects to replace `objects`
#    Example:
#        ```python
#        >>> import torch
#        >>> from accelerate.utils import release_memory
#        >>> a = torch.ones(1000, 1000).cuda()
#        >>> b = torch.ones(1000, 1000).cuda()
#        >>> a, b = release_memory(a, b)
#        ```
#    """
#    if not isinstance(objects, list):
#        objects = list(objects)
#    for i in range(len(objects)):
#        objects[i] = None
#    clear_device_cache(garbage_collection=True)
#    return objects
## %%
#import torch
#
#def run(use_checkpoint):
#    B, T, D = 64, 1024, 64
#    device = "cuda:0"
#    x = torch.randn(B, T, D, device=device, requires_grad=True)
#    for num_iters in range(1, 10):
#        clear_device_cache()
#        model = pRNN(D, num_layers=2, batch_first=True, num_iters=num_iters, method="minrnn", parallel=True).to(device)
#        model.checkpoint = use_checkpoint
#        y = model(x)
#        memory_used = torch.cuda.memory_allocated() / 1e9
#        print(f"Memory used after {num_iters} iterations: {memory_used:.2f} GB")
#        y.pow(2).sum().backward()
#        print(f"Memory used after {num_iters} iterations and backward: {memory_used:.2f} GB")
#        release_memory(model)
## %%
#run(use_checkpoint=False)
## %%
#run(use_checkpoint=True)
## %%