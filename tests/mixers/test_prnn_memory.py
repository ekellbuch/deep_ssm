"""
Compare memory
"""
import torch
import torch.nn as nn
from deep_ssm.mixers.prnn import pRNN
import pytest
import numpy as np

# Detect the appropriate device for execution
try:
    import torch_xla.core.xla_model as xm
    DEVICE = xm.xla_device()  # TPU
    atol = 1e-5
    rtol = 1e-6
except:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")  # macOS Metal backend
        atol = 1e-5
        rtol = 1e-6
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")  # GPU
        atol = 1e-3 # 1e-4
        rtol = 1e-5
    else:
        DEVICE = torch.device("cpu")  # CPU fallback
        atol = 1e-5
        rtol = 1e-6

torch.set_default_dtype(torch.float32)


from torch.profiler import profile, ProfilerActivity


def profile_memory(model, x, device="cuda"):
    model.to(device)
    x = x.to(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log")
    ) as prof:
        # Forward pass
        output, hidden = model(x)
        # Backward pass
        loss = output.mean()
        loss.backward()

    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))


class Arguments:
    def __init__(self, 
                input_size=10, 
                hidden_size=10, 
                num_layers=1, 
                batch_first=True, 
                dropout=0, 
                bidirectional=True, 
                num_iters=2,
                method="gru", 
                parallel=False,
                checkpoint=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.method = method
        self.parallel = parallel
        self.num_iters = num_iters
        self.checkpoint = checkpoint

import time

def measure_timing(model, x, device="cuda"):
    model.to(device)
    x = x.to(device)

    # Forward pass timing
    torch.cuda.synchronize()
    start = time.time()
    output, hidden = model(x)
    torch.cuda.synchronize()
    forward_time = time.time() - start

    # Backward pass timing
    loss = output.mean()
    torch.cuda.synchronize()
    start = time.time()
    loss.backward()
    torch.cuda.synchronize()
    backward_time = time.time() - start

    return forward_time, backward_time


def main():
    dim = 1
    bidirectional = False
    num_layers = 1
    batch = 1
    seqlen = 8
    num_iters = 2

    # Create an instance of Arguments
    cfg = Arguments(input_size=dim,
                    bidirectional=bidirectional,
                    num_layers=num_layers)

    D = 2 if cfg.bidirectional else 1
    hidden_size = cfg.hidden_size * D
    # Pass arguments to pRNN
    torch.manual_seed(42)
    np.random.seed(42)

    x = torch.randn(batch, seqlen, dim).to(DEVICE)

    # ---------
    # Get model
    torch.manual_seed(42)
    gru_model = nn.GRU(cfg.input_size,
                       cfg.hidden_size, 
                       cfg.num_layers, 
                       batch_first=cfg.batch_first, 
                       dropout=cfg.dropout, 
                       bidirectional=cfg.bidirectional).to(DEVICE)

    print("Profiling GRU:")
    profile_memory(gru_model, x, device=DEVICE)

    
    prnn = pRNN(**vars(cfg)).to(DEVICE)

    print("Profiling pRNN:")
    profile_memory(prnn, x, device=DEVICE)

    cfg2 = cfg
    cfg2.parallel = True
    cfg2.num_iters = num_iters
    prnn2 = pRNN(**vars(cfg2)).to(DEVICE)

    print("Profiling pRNN parallel:")
    profile_memory(prnn2, x, device=DEVICE)


if __name__ == "__main__":
    main()
