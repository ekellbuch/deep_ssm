"""
S5 forward pass duration:       0.004812 ms
S5 backward pass duration:      0.013665 ms
LSTM forward pass duration:     0.017165 ms
LSTM backward pass duration:    0.046078 ms
"""

#https://colab.research.google.com/drive/1ZGLiSoS7ijs8BrytYQbIp52WZ0PM3ZND?usp=sharing
import os
import torch
import torch.utils.benchmark as benchmark

from pathlib import Path
DEEP_SSM_PATH  = os.path.join(str(Path(__file__).parents[2]),"src")
print(DEEP_SSM_PATH)
import sys
sys.path.insert(0, DEEP_SSM_PATH)
from torch.profiler import profile, ProfilerActivity


from deep_ssm.models.s5_fjax.ssm import S5
# Profile and print FLOPs
from fvcore.nn import FlopCountAnalysis


# Doesn't work for arbitrary length
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L = 1024
B = 2
x_dim = 128
ssm_dim = 512


num_tests = 30
x = torch.randn(B, L, x_dim).to(device)

# Define a simple target tensor for the loss computation
target = torch.randn(B, L, x_dim).to(device)

# For S5 model
#from lightning.pytorch.utilities import measure_flops
# Calculate FLOPs using fvcore

def count_flops(model, x):
    flops = FlopCountAnalysis(model, x)
    return flops.total()

# For S5 model
s5_model = S5(x_dim, ssm_dim).to(device)
#s5_flops = count_flops(s5_model, x)
#print(f"S5 forward FLOPs: {s5_flops}")
#exit()
# TODO: for LSTM add decoder model instead of projection

t0_forward = benchmark.Timer(
    stmt='v, _ = lstm(x)',
    setup='lstm = torch.nn.LSTM(x_dim, ssm_dim, batch_first=True).to(device)',
    globals={'x': x, 'torch': torch, 'x_dim': x_dim, 'ssm_dim': ssm_dim, 'device':device})


t1_forward = benchmark.Timer(
    stmt='model(x)',
    setup='model = S5(x_dim, ssm_dim).to(device)',
    globals={'x': x, 'S5': S5, 'x_dim': x_dim, 'ssm_dim': ssm_dim, 'device':device})

# Backward pass timings
t0_backward = benchmark.Timer(
    stmt='''v, _ = lstm(x); loss = criterion(v, target); loss.backward(); lstm.zero_grad()''',
    setup='lstm = torch.nn.LSTM(x_dim, ssm_dim, batch_first=True, proj_size=x_dim).to(device);  criterion = torch.nn.MSELoss()',
    globals={'x': x, 'torch': torch, 'x_dim': x_dim, 'ssm_dim': ssm_dim, 'device':device, 'target': target})

t1_backward = benchmark.Timer(
    stmt='''v, state = model(x); loss = criterion(v, target); loss.backward(); model.zero_grad()''',
    setup='model = S5(x_dim, ssm_dim).to(device);  criterion = torch.nn.MSELoss()',
    globals={'x': x, 'S5': S5, 'x_dim': x_dim, 'ssm_dim': ssm_dim, 'device':device, 'target': target})

# Measure the times
time_lstm_forward = t0_forward.timeit(num_tests)
time_s5_forward = t1_forward.timeit(num_tests)
time_lstm_backward = t0_backward.timeit(num_tests)
time_s5_backward = t1_backward.timeit(num_tests)
# Print the results

print(f"LSTM forward pass duration:\t{time_lstm_forward.mean:.6f} ms")
print(f"S5 forward pass duration:\t{time_s5_forward.mean:.6f} ms")
print(f"LSTM backward pass duration:\t{time_lstm_backward.mean:.6f} ms")
print(f"S5 backward pass duration:\t{time_s5_backward.mean:.6f} ms")

# add flops:
