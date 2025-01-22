"""
heinsen.py - implement the complex valued scan from Heinsen
https://arxiv.org/abs/2311.06281

for consistency with the proger scan, make the input shape (B, D, T)

The heinsen scan is definitely faster, but less numerically stable
"""

import torch
import torch.nn.functional as F


def complex_log(float_input, eps=1e-6):
    eps = float_input.new_tensor(eps)
    real = float_input.abs().maximum(eps).log()
    imag = (float_input < 0).to(float_input.dtype) * torch.pi
    return torch.complex(real, imag)


def compute_in_parallel(coeffs, values):
    """
    Args:
      coeffs: (B,D,T)
      values: (B,D,T)
    """
    log_coeffs = complex_log(coeffs)  # (B,D,T)
    log_values = complex_log(values)  # (B,D,T)
    a_star = torch.cumsum(log_coeffs, dim=-1)  # (B,D,T)
    log_x0_plus_b_star = torch.logcumsumexp(
        log_values - a_star, dim=-1
    )  # eq (7) in paper
    log_x = a_star + log_x0_plus_b_star  # eq (1) in paper
    return torch.exp(log_x).real
