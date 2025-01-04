"""
Testing file for quasi_deer_torch
"""


import torch
from deep_ssm.mixers.prnn import quasi_deer_torch
import numpy as np
import pytest

# Device selection logic
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Metal Performance Shaders on macOS
elif torch.cuda.is_available():
    device = torch.device("cuda")  # CUDA on NVIDIA GPUs
else:
    device = torch.device("cpu")  # Fallback to CPU


def sample_fxn_torch(input, state):
    """
     Applies a simple transformation on the `state` tensor using a rotation matrix `A`
    and then a non-linear activation function (tanh).
    y = tanh(A @ x)

    Args:
       state (torch.Tensor): The input state vector or batch of state vectors (B, D).
       input: Unused argument in this function.

     Returns:
       torch.Tensor: The transformed state after applying the matrix and tanh activation.
    """
    device = state.device
    # Small rotation angle
    theta = torch.tensor(0.01, device=device)
    # Define a 2D rotation matrix based on theta
    A = torch.tensor(
        [[torch.cos(theta), torch.sin(theta)], [-torch.sin(theta), torch.cos(theta)]],
        device=device,
    )
    # Apply the rotation matrix and tanh activation
    # Note: Transpose `A` to align with PyTorch's row-major convention
    return torch.tanh(
        torch.matmul(state, A.T)
    )  # need to be very careful to handle batch dimension, (B,D)


def sample_diag_deriv_torch(input, state):
    """
    Computes the diagonal of the Jacobian matrix for `test_fxn_torch` with respect to `state`.

    Args:
      state (torch.Tensor): The input state vector or batch of state vectors (B, D).
      input: Unused argument in this function.

    Returns:
      torch.Tensor: The diagonal of the Jacobian matrix (B, D).
    """
    device = state.device
    theta = torch.tensor(0.01, device=device)
    # Define a 2D rotation matrix based on theta
    A = torch.tensor(
        [[torch.cos(theta), torch.sin(theta)], [-torch.sin(theta), torch.cos(theta)]],
        device=device,
    )
    # Compute activations by applying the rotation matrix
    activations = torch.matmul(state, A.T)  # (B,D)

    # Compute the diagonal entries directly using broadcasting
    activations_cosh_squared = torch.cosh(activations) ** 2  # (B,D)
    # Compute the diagonal entries of the derivative
    extra_diag = torch.diag_embed(1 / activations_cosh_squared)  # (B,D, D)
    # Multiply the derivative matrix with the transpose of A to compute diagonal entries
    # Use vmap to apply torch.diag for each batch element
    return torch.func.vmap(torch.diag)(torch.matmul(extra_diag, A.T))


# Sequential version of (quasi)deer that doesn't use parallel scan; avoids any implementational overhead and dependencies
def deer_step(f, J_f, input_seq, prev_iter_states):
    B, D, T = input_seq.shape
    prev_state = prev_iter_states[..., 0]  # (B,D)
    states = [prev_state]
    for t in range(1, T + 1):
        prev_state_prev_iter = prev_iter_states[..., t - 1]
        prev_state_diff = prev_state - prev_state_prev_iter
        curr_input = input_seq[..., t - 1]
        J_f_matrix = J_f(curr_input, prev_state_prev_iter)
        assert J_f_matrix.ndim == 2  # quasideer so we multiply by diagonal
        next_state = J_f_matrix * prev_state_diff + f(curr_input, prev_state_prev_iter)
        states.append(next_state)
        prev_state = next_state
    states = torch.stack(states, dim=-1)
    return states


def deer(f, J_f, input_seq, state_size, num_iters=1, initial_state=None):
    """
    Args
      f: forward function
      J_f: diagonal derivative
      input_seq: inputs
      state_size: D
      num_iters: number of iterations to run
      initial_state: (B,D)
    """
    B, _, T = input_seq.shape
    states = torch.zeros(
        B, state_size, T + 1, device=input_seq.device, dtype=input_seq.dtype
    )
    # set the inital state
    if initial_state is not None:
        states[..., 0] = initial_state

    for _ in range(num_iters):
        states = deer_step(f, J_f, input_seq, states)
    return states[..., 1:]


seqlens = [10000]
batch_sizes = [2] 
d_inputs = [2]
num_iters = [2, 5, 10] 
dtypes = [torch.float32]
seeds = [0] 

@pytest.mark.parametrize("seqlen", seqlens)
@pytest.mark.parametrize("batch_size", batch_sizes)
@pytest.mark.parametrize("d_input", d_inputs)
@pytest.mark.parametrize("num_iters", num_iters)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("seed", seeds)
def test_match_outputs(seqlen, batch_size, d_input, num_iters, dtype, seed):
    print(f"device is {device}")
    atol=1e-3
    torch.manual_seed(seed)

    D = 2
    initial_state_torch = torch.randn((batch_size, D), device=device, dtype=dtype)
    states_guess_torch = torch.zeros(
        (batch_size, D, seqlen), device=device, dtype=dtype
    )
    inputs_torch = torch.zeros(batch_size, d_input, seqlen, device=device, dtype=dtype)

    torch_out = quasi_deer_torch(
        sample_fxn_torch,
        sample_diag_deriv_torch,
        initial_state_torch,
        states_guess_torch,
        inputs_torch,
        num_iters=num_iters,
    )

    check_out = deer(
        sample_fxn_torch,
        sample_diag_deriv_torch,
        inputs_torch,
        D,
        num_iters=num_iters,
        initial_state=initial_state_torch,
    )

    assert torch.allclose(torch_out, check_out, atol=atol), f"the difference is {torch.max(torch.abs(torch_out - check_out))} \n with batch_size {batch_size} \n num_iters {num_iters} \n dtype {dtype} \n seed {seed}"
