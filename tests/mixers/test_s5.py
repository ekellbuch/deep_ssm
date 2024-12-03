"""
Compare jax and torch versions
"""
import pytest
import torch
import jax
import jax.numpy as jnp
from typing import Tuple
from deep_ssm.mixers.s5_fjax.jax_func import associative_scan

torch.autograd.set_detect_anomaly(True)

parallel_scan = jax.lax.associative_scan

def discretize_jax(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
    Args:
        Lambda (complex64): diagonal state matrix (P,)
        B_tilde (complex64): input matrix (P, H)
        Delta (float32): discretization step sizes (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)"""
    Identity = jnp.ones(Lambda.shape[0])
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


def binary_operator_jax(element_i, element_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        element_i: tuple containing A_i and Bu_i at position i (P,), (P,)
        element_j: tuple containing A_j and Bu_j at position j (P,), (P,)
    Returns:
        new element ( A_out, Bu_out ) """
    A_i, Bu_i = element_i
    A_j, Bu_j = element_j
    return A_j * A_i, A_j * Bu_i + Bu_j


def apply_ssm_jax(Lambda_bar, B_bar, C_tilde, D, input_sequence):
    """ Compute the LxH output of discretized SSM given an LxH input.
    Args:
        Lambda_bar (complex64): discretized diagonal state matrix (P,)
        B_bar (complex64): discretized input matrix (P, H)
        C_tilde (complex64): output matrix (H, P)
        D (float32): feedthrough matrix (H,)
        input_sequence (float32): input sequence of features (L, H)
    Returns:
        ys (float32): the SSM outputs (S5 layer preactivations) (L, H) """
    # Prepare elements required to initialize parallel scan
    Lambda_elements = jnp.repeat(Lambda_bar[None, ...], input_sequence.shape[0], axis=0)
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)
    elements = (Lambda_elements, Bu_elements) # (L, P), (L, P)

    # Compute latent state sequence given input sequence using parallel scan
    new_gates, xs = parallel_scan(binary_operator_jax, elements) # (L, P)

    # Compute SSM output sequence
    ys = jax.vmap(lambda x, u: (C_tilde @ x + D * u).real)(xs, input_sequence)
    return ys, xs[-1]


def torch_associative_scan(gates, tokens):
    """
    PyTorch equivalent of a parallel scan for linear recurrence using gates and tokens.

    Args:
        gates: Complex tensor representing the gates of shape (batch_size, seq_len, dim).
        tokens: Complex tensor representing the tokens of shape (batch_size, seq_len, dim).

    Returns:
        A tuple of tensors (scanned_gates, scanned_tokens) where:
            scanned_gates: The cumulative product of gates.
            scanned_tokens: The cumulative result of applying the recurrence relation.

    """
    seq_len = gates.shape[-2]

    # Initialize outputs
    scanned_gates = torch.zeros_like(gates)
    scanned_tokens = torch.zeros_like(tokens)

    # Set the initial values
    scanned_gates[..., 0, :] = gates[..., 0, :]
    scanned_tokens[..., 0, :] = tokens[..., 0, :]

    # Temporary tensors for computation
    current_gates = gates[..., 0, :]
    current_tokens = tokens[..., 0, :]

    for t in range(1, seq_len):
        # Compute new values
        current_gates = gates[..., t, :] * current_gates
        current_tokens = gates[..., t, :] * current_tokens + tokens[..., t, :]

        # Assign to output tensors
        scanned_gates[..., t, :] = current_gates
        scanned_tokens[..., t, :] = current_tokens

    return scanned_gates, scanned_tokens


def apply_ssm_jax_naive(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    # Prepare matrices for the sequential scan:
    # x_{t+1} = A x_t + B u_t
    # y = C x_{t+1}

    L, H = input_sequence.shape
    P = Lambda_bar.shape[0]

    # Sequential scan (naive implementation)
    xs = jnp.zeros((L, P), dtype=Lambda_bar.dtype)
    x = jnp.zeros(P, dtype=Lambda_bar.dtype) # P

    # For each timestep:
    for t in range(input_sequence.shape[0]):
      x = Lambda_bar * x + B_bar @ input_sequence[t]
      xs = xs.at[t].set(x)

    # Handle bidirectional case
    if bidirectional:
      raise  NotImplementedError("")
    # Apply the output transformation
    if conj_sym:
        ys = jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)
    else:
        ys = jax.vmap(lambda x: (C_tilde @ x).real)(xs)

    return ys


def apply_S5_layer_jax(params, input_sequence):
    """ Computes LxH output sequence of an S5 layer given LxH input sequence.
    Args:
        params: tuple of the continuous time SSM parameters
        input_sequence: input sequence of features (L, H)
    Returns:
        The S5 layer output sequence (L, H) """
    Lambda, B_tilde, C_tilde, D, log_Delta = params
    Lambda_bar, B_bar = discretize_jax(Lambda, B_tilde, jnp.exp(log_Delta))
    preactivations, _ = apply_ssm_jax(Lambda_bar, B_bar, C_tilde, D, input_sequence)
    return jax.nn.gelu(preactivations)


def batch_apply_S5_layer_jax(params, input_sequences):
    """ Computes BxLxH output sequence of an S5 layer given BxLxH input sequence.
    Args:
        params: tuple of the continuous time SSM parameters
        input_sequences: batch of input feature sequences (B, L ,H)
    Returns:
        Batch of S5 layer output sequences (B, L, H) """
    return jax.vmap(apply_S5_layer_jax, in_axes=(None, 0))(params, input_sequences)


@torch.jit.script
def binary_operator(
  q_i: Tuple[torch.Tensor, torch.Tensor], q_j: Tuple[torch.Tensor, torch.Tensor]
):
  """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
  Args:
      q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
      q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
  Returns:
      new element ( A_out, Bu_out )
  """
  A_i, b_i = q_i
  A_j, b_j = q_j

  # Ensure tensors are contiguous
  #A_i = A_i.contiguous()
  #A_j = A_j.contiguous()
  #b_i = b_i.contiguous()
  #b_j = b_j.contiguous()
  # return A_j * A_i, A_j * b_i + b_j

  #print('A_i', A_i.stride())
  #print('A_j', A_j.stride())
  return A_j * A_i,  A_j * b_i + b_j #torch.addcmul(b_j, A_j, b_i)


def discretize(Lambda: torch.Tensor,
                        B_tilde: torch.Tensor,
                        Delta: torch.Tensor,
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
  """Discretize a diagonalized, continuous-time linear SSM
  using bilinear transform method.
  Args:
      Lambda (complex64): diagonal state matrix              (P,)
          TensorType["num_states"]
      B_tilde (complex64): input matrix                      (P, H)
          TensorType["num_states", "num_features"]
      Delta (float32): discretization step sizes             (P,)
          TensorType["num_states"]
  Returns:
      discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
        Tuple[TensorType["num_states"], TensorType["num_states", "num_features"]]
  """
  Identity = torch.ones_like(Lambda)
  Lambda_bar = torch.exp(Lambda * Delta)
  B_bar = (1/ Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
  return Lambda_bar, B_bar


def apply_ssm(
  Lambda_bars: torch.Tensor,
  B_bars: torch.Tensor,
  C_tilde: torch.Tensor,
  D: torch.Tensor,
  input_sequence: torch.Tensor,
  conj_sym: bool = False,
  bidirectional: bool = False,
) -> Tuple[torch.Tensor,torch.Tensor]:
  """
  Apply a linear state-space model to an input sequence x_t and return cs:
    x_{t+1} = A x_t + B u
    cs: C_tilde x{t+1}

  :param Lambda_bars: diagonal state matrix: TensorType["num_states"]
  :param B_bars: input matrix: TensorType["num_states", "num_features"]
  :param C_tilde: output matrix: TensorType["num_features", "num_states"]
  :param input_sequence:TensorType["seq_length", "num_features"]
  :param prev_state:TensorType["num_states"]
  :param conj_sym:
  :param bidirectional:
  :return:  y, state:
    TensorType["seq_length", "num_features"], TensorType["num_states"]
  """
  # Cast to correct complex type
  cinput_sequence = input_sequence.type(Lambda_bars.dtype)

  # compute Bu elements
  Bu_elements = torch.vmap(lambda u: B_bars @ u, in_dims=0)(cinput_sequence)
  
  if Lambda_bars.ndim == 1:  # Repeat for associative_scan
    Lambda_bars = Lambda_bars.tile(input_sequence.shape[0], 1)

  #Lambda_bars[0] = Lambda_bars[0] * prev_state
  # compute state sequence using associative scan: x_{t+1} = A x_t + B u

  #new_gates, xs = associative_scan(binary_operator, (Lambda_bars, Bu_elements))
  new_gates, xs = torch_associative_scan(Lambda_bars, Bu_elements)
  
  if bidirectional:
    #_, xs2 = associative_scan(Lambda_bars, Bu_element), reverse=True
    #)
    #xs = torch.cat((xs, xs2), dim=-1)
    raise NotImplementedError("Bidirectional SSM not implemented")
  # TODO: the last element of xs (non-bidir) is the hidden state for bidir flag it!

  # compute SSM output sequence y = C_tilde x{t+1}
  if conj_sym:
    y = torch.vmap(lambda x, u: 2 * (C_tilde @ x + D * u).real)(xs, cinput_sequence)
  else:
    y = torch.vmap(lambda x, u: (C_tilde @x + D * u).real)(xs, cinput_sequence)

  if bidirectional:
    return y, xs[-1][:Lambda_bars.shape[-1]]
  else:
    return y, xs[-1]


def apply_ssm_broken(Lambda_bar, B_bar, C_tilde, D, input_sequence):
    """
    Compute the LxH output of discretized SSM given an LxH input.
    Args:
        Lambda_bar (complex64): discretized diagonal state matrix (P,)
        B_bar (complex64): discretized input matrix (P, H)
        C_tilde (complex64): output matrix (H, P)
        D (float32): feedthrough matrix (H,)
        input_sequence (float32): input sequence of features (L, H)
    Returns:
        ys (float32): the SSM outputs (S5 layer preactivations) (L, H)
    """
    cinput_sequence = input_sequence.type(Lambda_bar.dtype)

    L, H = cinput_sequence.shape  # Sequence length and input feature dimension
    P = Lambda_bar.shape[0]     # Number of states

    # Initialize latent state sequence (x_t)
    xs = torch.zeros(L, P, dtype=Lambda_bar.dtype, device=input_sequence.device)
    
    # Compute the first latent state (initial condition)
    Bu_elements = B_bar @ cinput_sequence.T  # Shape: (P, L)
    Bu_elements = Bu_elements.T  # Shape: (L, P)

    # Iteratively compute latent state sequence
    for t in range(1, L):
        xs[t] = Lambda_bar * xs[t - 1] + Bu_elements[t]

    # Compute the output sequence
    ys = torch.empty(L, H, dtype=input_sequence.dtype, device=input_sequence.device)
    for t in range(L):
        ys[t] = (C_tilde @ xs[t] + D * input_sequence[t]).real

    return ys, xs[-1]


def apply_S5_layer(params, input_sequence):
    """ Computes LxH output sequence of an S5 layer given LxH input sequence.
    Args:
        params: tuple of the continuous time SSM parameters
        input_sequence: input sequence of features (L, H)
    Returns:
        The S5 layer output sequence (L, H) """
    Lambda, B_tilde, C_tilde, D, log_Delta = params
    nonlin_ = torch.nn.GELU()
    Lambda_bar, B_bar = discretize(Lambda, B_tilde, torch.exp(log_Delta))
    preactivations, _ = apply_ssm(Lambda_bar, B_bar, C_tilde, D, input_sequence)
    return nonlin_(preactivations)


def batch_apply_S5_layer(params, input_sequences):
    """ Computes BxLxH output sequence of an S5 layer given BxLxH input sequence.
    Args:
        params: tuple of the continuous time SSM parameters
        input_sequences: batch of input feature sequences (B, L ,H)
    Returns:
        Batch of S5 layer output sequences (B, L, H) """
    return torch.vmap(apply_S5_layer, in_dims=(None, 0))(params, input_sequences)


seqlens = [4, 16]#[4, 16]# [4, 32, 64]
num_statess = [2]#[4, 16, 32]
d_models = [2]#[4]


@pytest.mark.parametrize("seqlen", seqlens)
@pytest.mark.parametrize("num_states", num_statess)
@pytest.mark.parametrize("d_model", d_models)
def test_match_output(seqlen,num_states,d_model):
    # Example input parameters
    P = num_states  # Number of states
    H = d_model  # Input and output feature dimensions
    L = seqlen  # Sequence length
    B = 1  # Batch size
    torch.manual_seed(42)
    dtype = torch.float32
    Lambda = torch.complex(torch.randn(P, dtype=dtype), torch.randn(P, dtype=dtype))
    B_tilde = torch.complex(torch.randn(P, H, dtype=dtype), torch.randn(P, H, dtype=dtype))
    C_tilde = torch.complex(torch.randn(H, P, dtype=dtype), torch.randn(H, P, dtype=dtype))
    D = torch.randn(H, dtype=dtype)
    log_Delta = torch.randn(P, dtype=dtype)

    params = (Lambda, B_tilde, C_tilde, D, log_Delta)
    params = [param.requires_grad_(True) for param in params]
    input_sequences = torch.randn(B, L, H, dtype=dtype)


    # Now let's compare the jax and torch versions
    dtype_jax= jnp.complex64
    Lambda_jax = jnp.array(Lambda.detach().numpy(), dtype=dtype_jax)
    B_tilde_jax = jnp.array(B_tilde.detach().numpy(), dtype=dtype_jax)
    C_tilde_jax = jnp.array(C_tilde.detach().numpy(), dtype=dtype_jax)
    D_jax = jnp.array(D.detach().numpy())
    log_Delta_jax = jnp.array(log_Delta.detach().numpy())

    assert jnp.allclose(Lambda.detach().numpy(), Lambda_jax, atol=1e-6)
    assert jnp.allclose(B_tilde.detach().numpy(), B_tilde_jax, atol=1e-6)
    assert jnp.allclose(C_tilde.detach().numpy(), C_tilde_jax, atol=1e-6)
    assert jnp.allclose(D.detach().numpy(), D_jax, atol=1e-6)
    assert jnp.allclose(log_Delta.detach().numpy(), log_Delta_jax, atol=1e-6)

    params_jax = (Lambda_jax, B_tilde_jax, C_tilde_jax, D_jax, log_Delta_jax)
    input_sequences_jax = jnp.array(input_sequences.detach().numpy())

    # compare discretization
    Lambda_bar_torch, B_bar_torch = discretize(Lambda, B_tilde, log_Delta)
    Lambda_bar_jax, B_bar_jax = discretize_jax(Lambda_jax, B_tilde_jax, log_Delta_jax)

    assert jnp.allclose(Lambda_bar_torch.detach().numpy(), Lambda_bar_jax, atol=1e-5), print("Failed discretization")
    assert jnp.allclose(B_bar_torch.detach().numpy(), B_bar_jax, atol=1e-5), print("Failed discretization")

    # compare apply ssm
    torch_output, torch2 = apply_ssm(Lambda_bar_torch, B_bar_torch, C_tilde, D, input_sequences[0])
    jax_output, jaz2 = apply_ssm_jax(Lambda_bar_jax, B_bar_jax, C_tilde_jax, D_jax, input_sequences_jax[0])
    assert jnp.allclose(torch_output.detach().cpu().numpy(), jax_output, atol=1e-6), "Outputs of apply ssm not match"
    assert jnp.allclose(torch2.detach().cpu().numpy(), jaz2, atol=1e-6), "Outputs of apply ssm not match"


    # Single sequence example
    output_sequence = batch_apply_S5_layer(params, input_sequences)
    #print("Single sequence output:")
    assert output_sequence.shape == (B, L, H)


    output_sequence_jax = batch_apply_S5_layer_jax(params_jax, input_sequences_jax)
    assert output_sequence.shape == output_sequence_jax.shape
    assert jnp.allclose(output_sequence.cpu().detach().numpy(), output_sequence_jax, atol=1e-3)

    # Test backpropagation
    # PyTorch backpropagation
    output_sequence.mean().backward()
    torch_grads = [param.grad for param in params]

    # JAX backpropagation
    def loss_fn_jax(params_jax, input_sequences_jax):
        output_sequence_jax = batch_apply_S5_layer_jax(params_jax, input_sequences_jax)
        return output_sequence_jax.mean()

    jax_grads = jax.grad(loss_fn_jax)(params_jax, input_sequences_jax)

    # Compare gradients
    for torch_grad, jax_grad in zip(torch_grads, jax_grads):
        # Convert PyTorch gradients to numpy
        torch_grad_np = torch_grad.cpu().detach().numpy()
        if jnp.iscomplexobj(jax_grad) or jnp.iscomplexobj(torch_grad_np):
            # pytorch uses the conjugate transpose for complex gradients: (https://github.com/jax-ml/jax/issues/9110)
            # If complex, compare real and imaginary parts separately
            assert jnp.allclose(torch_grad_np.real, jax_grad.real, atol=1e-4, rtol=1e-4), \
                f"Real parts of gradients do not match: {torch_grad_np.real} vs {jax_grad.real}"
            assert jnp.allclose(-1*torch_grad_np.imag, jax_grad.imag, atol=1e-4, rtol=1e-4), \
                f"Imaginary parts of gradients do not match: {-1*torch_grad_np.imag} vs {jax_grad.imag}"
        else:
            # If real, compare directly
            assert jnp.allclose(torch_grad_np, jax_grad, atol=1e-4, rtol=1e-4), \
                f"Gradients do not match: {torch_grad_np} vs {jax_grad}"

