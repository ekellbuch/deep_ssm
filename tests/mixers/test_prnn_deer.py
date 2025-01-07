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


def buildW():
    theta = torch.tensor(0.01)
    # Define a 2D rotation matrix based on theta
    A = torch.tensor(
        [[torch.cos(theta), torch.sin(theta)], [-torch.sin(theta), torch.cos(theta)]],
    )
    return A


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
    A = buildW()
    A = A.to(device=device)
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

def sample_full_deriv_torch(input, state):
    """
    Computes the full Jacobian matrix for `sample_fxn_torch` with respect to `state`.

    Args:
      state (torch.Tensor): The input state vector or batch of state vectors (B, D).
      input: Unused argument in this function.

    Returns:
      torch.Tensor: The full Jacobian matrix (B, D, D).
    """
    device = state.device
    theta = torch.tensor(0.01, device=device)
    # Define a 2D rotation matrix based on theta
    A = torch.tensor(
        [[torch.cos(theta), torch.sin(theta)], [-torch.sin(theta), torch.cos(theta)]],
        device=device,
    )

    # Compute activations by applying the rotation matrix
    activations = torch.matmul(state, A.T)  # (B, D)

    # Compute the derivative of tanh (1 - tanh^2) element-wise
    activations_tanh = torch.tanh(activations)  # (B, D)
    
    if device.type == "mps":
        tanh_derivative = 1 - activations_tanh.to("cpu") ** 2  # (B, D)
        tanh_derivative = tanh_derivative.to(activations.device)
    else:
        tanh_derivative = 1 - activations_tanh ** 2  # (B, D)

    # Scale each row of A.T by tanh_derivative
    # (B, D, D) where each (D, D) Jacobian is A.T scaled row-wise
    J_f = torch.einsum('bi,ij->bij', tanh_derivative, A.T)

    return J_f


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


def stabilized_diagonalization(jacobian, epsilon=1e-10):
    """
    Perform numerically stable diagonalization with conditioning checks.
    
    Args:
        jacobian: Input matrix to diagonalize
        epsilon: Small value for numerical stability
    
    Returns:
        P: Eigenvector matrix
        P_inv: Inverse of eigenvector matrix
        eigenvalues: Diagonal matrix of eigenvalues
    """
    # Compute eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eig(jacobian)
    
    # Check condition number
    condition_number = torch.linalg.cond(eigenvectors)
    if condition_number > 1e6:  # Threshold for poor conditioning
        # Add small regularization to improve conditioning
        reg_matrix = torch.eye(jacobian.shape[0], device=jacobian.device) * epsilon
        eigenvalues, eigenvectors = torch.linalg.eig(jacobian + reg_matrix)
    
    # Compute inverse more carefully
    if eigenvectors.device.type == "mps":
        P_inv = torch.linalg.inv(eigenvectors.cpu()).to(eigenvectors.device)
    else:
        P_inv = torch.linalg.inv(eigenvectors)

    # Verify decomposition quality
    reconstruction = eigenvectors @ torch.diag(eigenvalues) @ P_inv

    if eigenvectors.device.type == "mps":
        jac_cpu = jacobian.cpu()
        reconstruction_error = torch.norm(reconstruction.cpu() - jac_cpu) / torch.norm(jac_cpu)
        reconstruction_error = reconstruction_error.to(device=jacobian.device)
    else:
        reconstruction_error = torch.norm(reconstruction - jacobian) / torch.norm(jacobian)

    if reconstruction_error > 1e-6:
        print(f"Warning: Large reconstruction error: {reconstruction_error}")
        
    return eigenvectors, P_inv, eigenvalues


seqlens = [8]
batch_sizes = [1] 
d_inputs = [2]
num_iters = [3000] 
dtypes = [torch.float32]
seeds = [42] 

@pytest.mark.parametrize("seqlen", seqlens)
@pytest.mark.parametrize("batch_size", batch_sizes)
@pytest.mark.parametrize("d_input", d_inputs)
@pytest.mark.parametrize("num_iters", num_iters)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("seed", seeds)
def test_complex_outputs(seqlen, batch_size, d_input, num_iters, dtype, seed):
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

    # Get the Jacobian for f(x) = y = tanh(W @ x)
    # Jacobian wrt to x:  J_f = diag(1-tanh^2 (W x))⋅W
    # Let' say we start with x_0=0 ->  J_f (x_0) = W

    # Now Let's consider a complex reparameterization:
    # J_f(x_0) = P_0 V_0 P_0 ^-1 = W
    # transform x_t into the eigenbasis, we have:
    # z_t = P0^-1 x_t
    # z_{t+1} = P_0 ^-1 f(P_0 z_t) = g(z_t)
    # J_g(z_t) = P_0 ^ -1 J_f P_0
    # J_g(z_t) = P_0 ^-1 diag (tanh' (P_0 z_t)) P_0 V_0 

    jacobian = buildW().to(device)
    #jacobian = torch.autograd.functional.jacobian(lambda x, h: sample_fxn_torch(x, h)[1], (inputs_torch[...,0], initial_state_torch[0]))
    #eigenvalues, eigenvectors = torch.linalg.eig(jacobian)
    #P = eigenvectors
    # because torch doesn't support inv of complex number:
    #P_inv = torch.linalg.inv(eigenvectors.cpu())
    #P_inv = P_inv.to(P.device)
    P, P_inv, eigenvalues = stabilized_diagonalization(jacobian)

    def g_fxn_torch(u, z):
        return sample_fxn_torch(u, z @ P) @ P_inv
    
    def g_diag_deriv_torch(input, state):
        # Transform the hidden state to Z-space:
        # Let's just do naive computation:
        # z_hx = torch.matmul(state, P_inv)
        # diag_jacobian = sample_diag_deriv_torch(input, z_hx)
        J_f = sample_full_deriv_torch(input, state)

        transform_single = lambda J_f_row: torch.diag(P_inv @ (J_f_row @ P))

        diag_jacobian= torch.vmap(transform_single)(J_f)
        return diag_jacobian

    z_initial_state_torch = initial_state_torch.to(P_inv.dtype) @ P_inv 
    states_guess_torch = states_guess_torch.to(P_inv.dtype)
    inputs_torch = inputs_torch.to(P_inv.dtype)

    z_out = quasi_deer_torch(
        g_fxn_torch,
        g_diag_deriv_torch,
        z_initial_state_torch,
        states_guess_torch,
        inputs_torch,
        num_iters=num_iters,
    )
    # TODO: DEER is not actually DEER here
    check_out = P @ z_out 
    assert torch.allclose(torch_out, check_out.real, atol=atol), f"the difference is {torch.max(torch.abs(torch_out - check_out))} "
