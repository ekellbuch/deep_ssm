"""
Compare memory and timing:
"""
import torch
import torch.nn as nn
from deep_ssm.mixers.prnn import pRNN
import pytest
import numpy as np
import gc
from torch.profiler import profile, ProfilerActivity
import time

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



def profile_memory(model, x, device="cuda"):
    model.to(device)
    x = x.to(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        #record_shapes=True,
        #on_trace_ready=torch.profiler.tensorboard_trace_handler("./log")
    ) as prof:
        # Forward pass
        output, hidden = model(x)
        # Backward pass
        loss = output.mean()
        loss.backward()

    #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    return "Timing profiling done."

def clear_memory():
    gc.collect()  # Free CPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear PyTorch's memory cache
        torch.cuda.reset_peak_memory_stats()  # Reset peak memory tracking
        torch.cuda.synchronize()  # Ensure all GPU operations are complete
    

def measure_gpu_memory(model, x, device="cuda"):
    model.to(device)
    x = x.to(device)

    # Clear any existing cache
    torch.cuda.empty_cache()

    # Measure memory before forward pass
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    before_fwd = torch.cuda.memory_allocated(device)

    # Forward pass
    output, hidden = model(x)
    torch.cuda.synchronize()
    after_fwd = torch.cuda.memory_allocated(device)
    peak_fwd = torch.cuda.max_memory_allocated(device)

    # Backward pass
    loss = output.mean()
    loss.backward()
    torch.cuda.synchronize()
    after_bwd = torch.cuda.memory_allocated(device)
    peak_bwd = torch.cuda.max_memory_allocated(device)

    # Results
    return {
        "memory_before_fwd": before_fwd,
        "memory_after_fwd": after_fwd,
        "peak_memory_fwd": peak_fwd,
        "memory_after_bwd": after_bwd,
        "peak_memory_bwd": peak_bwd,
    }


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



def main():
    dim = 2
    bidirectional = True
    num_layers = 2
    batch = 2
    seqlen = 32
    num_iters = 10

    # Create an instance of Arguments
    cfg = Arguments(input_size=dim,
                    bidirectional=bidirectional,
                    num_layers=num_layers)

    D = 2 if cfg.bidirectional else 1
    hidden_size = cfg.hidden_size * D
    # Pass arguments to pRNN
    torch.manual_seed(42)
    np.random.seed(42)

    x = torch.randn(batch, seqlen, dim)#.to(DEVICE)

    if DEVICE.type in ("cpu","mps"):
        f_eval = profile_memory
    else:
        #f_eval = measure_gpu_memory
        f_eval = measure_timing
    print(f_eval)
    # ---------
    # Get model
    torch.manual_seed(42)
    
    gru_model = nn.GRU(cfg.input_size,
                       cfg.hidden_size, 
                       cfg.num_layers, 
                       batch_first=cfg.batch_first, 
                       dropout=cfg.dropout, 
                       bidirectional=cfg.bidirectional)

    print("\n Profiling GRU:")
    memory_usage = f_eval(gru_model, x, device=DEVICE)
    print(memory_usage)
    del gru_model

    clear_memory()
    """
    prnn_model = pRNN(**vars(cfg))

    print("\n Profiling pRNN:")
    memory_usage = f_eval(prnn_model, x, device=DEVICE)
    print(memory_usage)
    del prnn_model
    clear_memory()
    """
    cfg2 = cfg
    cfg2.parallel = True
    cfg2.num_iters = num_iters
    prnn2 = pRNN(**vars(cfg2))
    print("\n Profiling pRNN parallel:")
    memory_usage = f_eval(prnn2, x, device=DEVICE)
    print(memory_usage)


if __name__ == "__main__":
    main()
