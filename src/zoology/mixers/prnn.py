# %%
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