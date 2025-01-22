"""
Testing file for quasi_deer_torch and deer_torch
"""

import torch
from deep_ssm.mixers.min_gru import CausalConv1d
import numpy as np
import pytest
import wandb

# Device selection logic
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Metal Performance Shaders on macOS
elif torch.cuda.is_available():
    device = torch.device("cuda")  # CUDA on NVIDIA GPUs
else:
    device = torch.device("cpu")  # Fallback to CPU


def test_causal_convolution():
    # Define parameters
    D = 3  # Number of features (input channels)
    kernel_size = 5  # Kernel size for convolution
    batch_size = 2
    time_steps = 10

    # Create an instance of the CausalConv1d class
    causal_conv = CausalConv1d(D, kernel_size)
    causal_conv.conv.weight.data.fill_(1.0)  # Set weights to 1 for simplicity
    causal_conv.conv.bias.data.fill_(0.0)  # Set bias to 0

    # Create input with a known pattern: zeros after time step 5
    input_tensor = torch.zeros(batch_size, time_steps, D)
    input_tensor[:, :5, :] = torch.arange(1, 6).view(
        -1, 1
    )  # Non-zero values in first 5 steps

    # Apply causal convolution
    output = causal_conv(input_tensor)

    # Convert output to numpy for easier inspection
    output_np = output.detach().numpy()

    # Verify causality: output at t depends only on inputs <= t
    for b in range(batch_size):
        for d in range(D):
            for t in range(time_steps):
                # Compute expected output: sum of kernel-size past inputs
                start_t = max(0, t - kernel_size + 1)
                expected_sum = input_tensor[b, start_t : t + 1, d].sum().item()
                assert np.isclose(output_np[b, t, d], expected_sum), (
                    f"Failed causality check at batch {b}, time {t}, feature {d}: "
                    f"Expected {expected_sum}, but got {output_np[b, t, d]}"
                )
