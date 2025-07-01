"""
Compare jax and torch S5 layer implementations.

An S5 layer is a linear recurrence with a diagonal state matrix.

The state matrix is diagonalized and discretized using the bilinear transform method.

The S5 layer is implemented in JAX and PyTorch.

The test compares the output of the S5 layer and the gradients computed by the backward pass.

"""
import pytest
import torch
import jax
import jax.numpy as jnp
from typing import Tuple
from deep_ssm.mixers.s5_fjax.jax_func import associative_scan
from deep_ssm.mixers.s5_fjax.ssm_init import make_DPLR_HiPPO
from deep_ssm.mixers.scan_utils import torch_associative_scanBTC, scan_btc_complex_autograd
from deep_ssm.mixers.scan_utils import scan_btc_complex_autograd as scan_tc_complex

try:
   import triton
   import triton.language as tl
   load_triton = True
except:
   load_triton = False

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["TRITON_INTERPRET"]="1"
os.environ["TRITON_LOG_LEVEL"] = "debug"

torch.autograd.set_detect_anomaly(True)

parallel_scan = jax.lax.associative_scan

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def to_jax(x):
    return x.cpu().detach().numpy()


def discretize_jax(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
    Args:
        Lambda (complex64): diagonal state matrix (P,)
        B_tilde (complex64): input matrix (P, H)
        Delta (float32): discretization step sizes (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)"""
    Identity = jnp.ones(Lambda.shape[0], dtype=Lambda.dtype)
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


@jax.vmap
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


def apply_ssm_jax(Lambda_bar, B_bar, C_tilde, D, input_sequence, conj_sym=False, bidirectional=False):
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
    Lambda_elements = Lambda_bar * jnp.ones((input_sequence.shape[0], Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)
    elements = (Lambda_elements, Bu_elements) # (L, P), (L, P)

    # Compute latent state sequence given input sequence using parallel scan
    _, xs = parallel_scan(binary_operator_jax, elements) # (L, P)

    if bidirectional:
       _, xs2 = parallel_scan(binary_operator_jax, elements, reverse=True) # (L, P)
       xs = jnp.concatenate([xs, xs2], axis=-1)

    # Compute SSM output sequence
    if conj_sym:
        ys = jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)
    else:
        ys = jax.vmap(lambda x: (C_tilde @ x).real)(xs) 
    
    Du = jax.vmap(lambda u: D * u)(input_sequence) # (L, H)
    ys = ys + Du
    if bidirectional:
        xs = xs[-1][:Lambda_bar.shape[-1]]
    else:
        xs = xs[-1]
    return ys, xs


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


def apply_S5_layer_jax(params, input_sequence, bidirectional=False):
    """ Computes LxH output sequence of an S5 layer given LxH input sequence.
    Args:
        params: tuple of the continuous time SSM parameters
        input_sequence: input sequence of features (L, H)
    Returns:
        The S5 layer output sequence (L, H) """
    Lambda, B_tilde, C_tilde, D, log_Delta = params
    Lambda_bar, B_bar = discretize_jax(Lambda, B_tilde, jnp.exp(log_Delta))
    preactivations, _ = apply_ssm_jax(Lambda_bar, B_bar, C_tilde, D, input_sequence, bidirectional=bidirectional)
    return jax.nn.gelu(preactivations)


def batch_apply_S5_layer_jax(params, input_sequences, bidirectional=False):
    """ Computes BxLxH output sequence of an S5 layer given BxLxH input sequence.
    Args:
        params: tuple of the continuous time SSM parameters
        input_sequences: batch of input feature sequences (B, L ,H)
    Returns:
        Batch of S5 layer output sequences (B, L, H) """
    return jax.vmap(lambda g, t: apply_S5_layer_jax(g, t, bidirectional=bidirectional), in_axes=(None, 0))(params, input_sequences)


# ----------------- PyTorch functions ----------------- 


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

# With this:
def pytorch_match_jax_complex_grad(C_tilde, x):
    """Force PyTorch to compute complex @ gradients like JAX."""
    # Split: (C_real + i*C_imag) @ (x_real + i*x_imag)
    # Real part = C_real @ x_real - C_imag @ x_imag
    real_part = torch.mv(C_tilde.real, x.real) - torch.mv(C_tilde.imag, x.imag)
    return real_part


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
  Lambda_bars = Lambda_bars * torch.ones((input_sequence.shape[0], Lambda_bars.shape[0]), device=Lambda_bars.device)
  cinput_sequence = input_sequence.type(Lambda_bars.dtype)
  seq_len = input_sequence.shape[0]
  # compute Bu elements
  Bu_elements = torch.vmap(lambda u: B_bars @ u, in_dims=0)(cinput_sequence) # (P, D ) [L, D] -> [L, P]
  
  #Lambda_elements = jnp.repeat(Lambda_bar[None, ...], input_sequence.shape[0], axis=0)
  if Lambda_bars.ndim == 1 :  # Repeat along L for associative_scan
    Lambda_bars = Lambda_bars.unsqueeze(0).expand(seq_len, -1) # (L, P)

  _, xs = scan_tc_complex(Lambda_bars, Bu_elements)

  if bidirectional:
    _, xs2 = scan_tc_complex(Lambda_bars, Bu_elements, reverse=True)
    xs = torch.cat([xs, xs2], dim=-1)

  # compute SSM output sequence y = C_tilde x{t+1}
  if conj_sym:
    y = torch.vmap(lambda x: 2 * pytorch_match_jax_complex_grad(C_tilde, x).real)(xs)
  else:
    y = torch.vmap(lambda x: (pytorch_match_jax_complex_grad(C_tilde, x)).real )(xs)
  
  Du = torch.vmap(lambda u: D * u)(cinput_sequence)
  y = y + Du.real

  if bidirectional:
    xs = xs[-1][:Lambda_bars.shape[-1]]
  else:
    xs = xs[-1]
  return y, xs


def apply_S5_layer(params, input_sequence, bidirectional=False):
    """ Computes LxH output sequence of an S5 layer given LxH input sequence.
    Args:
        params: tuple of the continuous time SSM parameters
        input_sequence: input sequence of features (L, H)
    Returns:
        The S5 layer output sequence (L, H) """
    Lambda, B_tilde, C_tilde, D, log_Delta = params
    nonlin_ = torch.nn.GELU()
    Lambda_bar, B_bar = discretize(Lambda, B_tilde, torch.exp(log_Delta))
    preactivations, _ = apply_ssm(Lambda_bar, B_bar, C_tilde, D, input_sequence, bidirectional=bidirectional)
    return nonlin_(preactivations)


def batch_apply_S5_layer(params, input_sequences, bidirectional=False):
    """ Computes BxLxH output sequence of an S5 layer given BxLxH input sequence.
    Args:
        params: tuple of the continuous time SSM parameters
        input_sequences: batch of input feature sequences (B, L ,H)
    Returns:
        Batch of S5 layer output sequences (B, L, H) """
    return torch.vmap(lambda g, t: apply_S5_layer(g, t, bidirectional=bidirectional), in_dims=(None, 0))(params, input_sequences)

def compare_parameters(params_torch, params_jax, atol=1e-3, rtol=1e-2):
    for ii, (param_torch, param_jax) in enumerate(zip(params_torch, params_jax)):
        assert jnp.allclose(to_jax(param_torch), param_jax, atol=atol, rtol=rtol), print(f"Parameters {ii}d not match: \n torch {param_torch} \n jax {param_jax}")
    return True

def func_compare_complex(var_torch, var_jax, atol=1e-6, rtol=1e-6, var_name="Variable"):
    # check variables match for real and complex parts
    var_torch_re = var_torch.real
    var_torch_im = var_torch.imag
    var_jax_re = var_jax.real
    var_jax_im = var_jax.imag
    assert jnp.allclose(var_torch_re, var_jax_re, atol=atol, rtol=rtol), f"{var_name} do not match: PyTorch {var_torch_re}, JAX {var_jax_re}"
    assert jnp.allclose(var_torch_im, var_jax_im, atol=atol, rtol=rtol), f"{var_name} do not match: PyTorch {var_torch_im}, JAX {var_jax_im}"

def torch_loss(x, Lre, Lim, other_params, bidirectional=False):
        Lam = torch.complex(Lre, Lim)
        params = (Lam, *other_params)
        y = batch_apply_S5_layer(params, x, bidirectional=bidirectional)      # (B,L,H) → real
        return y.real.mean()

def jax_loss(x, Lre, Lim, other_params, bidirectional=False):
    Lam = Lre + 1j * Lim
    params = (Lam, *other_params)
    y = batch_apply_S5_layer_jax(params, x, bidirectional=bidirectional)
    return y.real.mean()

def make_initial_params(P, H, dtype, device, complexity, dtype_complex, bidirectional=False):
    if complexity == "v0":
        Lambda, P_term, B_tilde, V, B_orig = make_DPLR_HiPPO(P)
        V = torch.from_numpy(V).to(dtype_complex).to(device)
        Vinv = V.conj().T
        # TODO: add blocks
        #Vinv = torch.block_diag(*([Vc] * 1))
        B_tilde = torch.empty(P, H, dtype=dtype_complex)
        torch.nn.init.xavier_uniform_(B_tilde)
        B_tilde = Vinv @ B_tilde.to(device)
        Lambda = torch.complex(torch.from_numpy(Lambda.real).to(dtype), torch.from_numpy(Lambda.imag).to(dtype)).to(dtype_complex).to(device)

        C_real = torch.empty(H, P, dtype=dtype)
        torch.nn.init.xavier_uniform_(C_real)
        C_imag = torch.empty(H, P, dtype=dtype)
        torch.nn.init.xavier_uniform_(C_imag)
        C_ = torch.complex(C_real, C_imag).to(dtype_complex).to(device)
        C_tilde = C_ @ V
        #C_tilde = torch.complex(C_init[..., 0], C_init[..., 1]).to(dtype_complex).to(device)
    elif complexity == "v1":
        Lambda = torch.complex(torch.randn(P, dtype=dtype), torch.zeros(P, dtype=dtype)).to(device)
        B_tilde = torch.complex(torch.randn(P, H, dtype=dtype), torch.zeros(P, H, dtype=dtype)).to(device) 
        C_tilde = torch.complex(torch.ones(H, P, dtype=dtype), torch.zeros(H, P, dtype=dtype)).to(device)
    elif complexity == "v2":
        Lambda = torch.complex(torch.randn(P, dtype=dtype), torch.randn(P, dtype=dtype)).to(device)
        B_tilde = torch.complex(torch.randn(P, H, dtype=dtype), torch.randn(P, H, dtype=dtype)).to(device)
        C_tilde = torch.complex(torch.ones(H, P, dtype=dtype), torch.ones(H, P, dtype=dtype)).to(device)
    else:
        raise ValueError(f"Complexity {complexity} not supported")
    
    # Normalize to avoid overflow
    Lambda = Lambda / (P*H)
    B_tilde = B_tilde / (P*H)
    C_tilde = C_tilde / (H*P)
    
    if bidirectional:
        C_tilde = torch.cat([C_tilde, C_tilde], dim=-1)
    return Lambda, B_tilde, C_tilde


seqlens = [4] # seqlen
num_statess = [2] # num_states
d_models = [2] # d_model
batch_dims = [2] # batch_dim
complexities = ["v0", "v1"]
bidirectionals = [False,True]

@pytest.mark.parametrize("seqlen", seqlens)
@pytest.mark.parametrize("num_states", num_statess)
@pytest.mark.parametrize("d_model", d_models)
@pytest.mark.parametrize("batch", batch_dims)
@pytest.mark.parametrize("complexity", complexities)
@pytest.mark.parametrize("bidirectional", bidirectionals)
def test_match_output(seqlen, num_states,d_model, batch, complexity, bidirectional):
    # Example input parameters
    P = num_states  # Number of states
    H = d_model  # Input and output feature dimensions
    L = seqlen  # Sequence length
    B = batch  # Batch size
    torch.manual_seed(42)
    dtype = torch.float32
    dtype_complex = torch.complex64

    if P > L:
        return
    # complex number tolerance
    atol = 1e-2  # absolute tolerance
    rtol = 1e-2  # max relative difference

    Lambda, B_tilde, C_tilde = make_initial_params(P, H, dtype, device, complexity, dtype_complex, bidirectional)

    #Lambda = torch.complex(torch.clip(Lambda.real, None, -1e-4), Lambda.imag)
    Lambda = Lambda / Lambda.abs().max()
    #B_tilde = B_tilde / B_tilde.abs().max()
    #C_tilde = C_tilde / C_tilde.abs().max()
    D = torch.randn(H, dtype=dtype).to(device)
    log_Delta = torch.randn(P, dtype=dtype).to(device)

    dtype_jax= jnp.complex64
    Lambda_jax = jnp.array(to_jax(Lambda), dtype=dtype_jax)
    B_tilde_jax = jnp.array(to_jax(B_tilde), dtype=dtype_jax)
    C_tilde_jax = jnp.array(to_jax(C_tilde), dtype=dtype_jax)
    D_jax = jnp.array(to_jax(D))
    log_Delta_jax = jnp.array(to_jax(log_Delta))

    # Define jax params   
    params = (Lambda, B_tilde, C_tilde, D, log_Delta)
    params = [param.requires_grad_(True) for param in params]
    input_sequences = torch.randn(B, L, H, dtype=dtype).to(device)

    params_jax = (Lambda_jax, B_tilde_jax, C_tilde_jax, D_jax, log_Delta_jax)
    input_sequences_jax = jnp.array(input_sequences.cpu().detach().numpy())

    compare_parameters(params, params_jax, atol=1e-6, rtol=1e-6)

    # -----------------------------
    # compare discretization
    Lambda_bar_torch, B_bar_torch = discretize(Lambda, B_tilde, log_Delta)
    Lambda_bar_jax, B_bar_jax = discretize_jax(Lambda_jax, B_tilde_jax, log_Delta_jax)

    assert jnp.allclose(to_jax(Lambda_bar_torch), Lambda_bar_jax, atol=1e-4), "Lambda_bar does not match"
    assert jnp.allclose(to_jax(B_bar_torch), B_bar_jax, atol=1e-4), "B_bar does not match"

    # -----------------------------
    # compare apply ssm
    """
    jax_output, jaz2 = apply_ssm_jax(Lambda_bar_jax, B_bar_jax, C_tilde_jax, D_jax, input_sequences_jax[0], bidirectional=bidirectional)

    torch_output, torch2 = apply_ssm(Lambda_bar_torch, B_bar_torch, C_tilde, D, input_sequences[0], bidirectional=bidirectional)

    assert jnp.allclose(to_jax(torch_output), jax_output, atol=1e-4), "Output does not match"
    assert jnp.allclose(to_jax(torch2), jaz2, atol=1e-4), "Output does not match"

    # Single sequence example
    for param in params:
        param.grad = None
    """
    output_sequence_jax = batch_apply_S5_layer_jax(params_jax, input_sequences_jax, bidirectional=bidirectional)

    output_sequence = batch_apply_S5_layer(params, input_sequences, bidirectional=bidirectional)
    #print("Single sequence output:")
    assert output_sequence.shape == (B, L, H)
    assert output_sequence.shape == output_sequence_jax.shape

    # Compare output
    assert jnp.allclose(output_sequence_jax.mean(), to_jax(output_sequence.mean()), atol=atol, rtol=rtol), "Loss of bacthed S5 layer do not match"

    # ----------------------------------------------------------------------------------------------
    # Test loss computation
    for param in params:
        if param.grad is not None:
            param.grad = None

    # --- PyTorch ----------------------------------------------------------
    # 1. Split Lambda into real & imag parts that require_grad = True
    Lambda_re = Lambda.real.clone().detach().requires_grad_(True)
    Lambda_im = Lambda.imag.clone().detach().requires_grad_(True)

    other_pt = (B_tilde, C_tilde, D, log_Delta)
    other_jx = (B_tilde_jax, C_tilde_jax, D_jax, log_Delta_jax)

    # ------------------------------------------------------------------
    # 3. Forward pass – PyTorch
    loss_pt = torch_loss(input_sequences, Lambda_re, Lambda_im, other_pt, bidirectional=bidirectional)

    # ------------------------------------------------------------------
    # 4. Forward pass – JAX
    Lre_jx = Lambda_jax.real
    Lim_jx = Lambda_jax.imag
    loss_jx = jax_loss(input_sequences_jax, Lre_jx, Lim_jx, other_jx, bidirectional=bidirectional)

    # ------------------------------------------------------------------
    # 5. Compare loss values
    assert jnp.allclose(to_jax(loss_pt), loss_jx, atol=atol, rtol=rtol), f"Loss values do not match: PyTorch {loss_pt.item():.6f}, JAX {loss_jx:.6f}"

    # 6. Compare torch gradients 
    # 6.1. PyTorch
    loss_pt.backward()
    grad_Lre_pt = Lambda_re.grad.clone().detach()
    grad_Lim_pt = Lambda_im.grad.clone().detach()

    # 6.2. JAX
    grads_jx = jax.grad(jax_loss, (1, 2))(input_sequences_jax, Lre_jx, Lim_jx, other_jx, bidirectional=bidirectional)
    grad_Lre_jx, grad_Lim_jx = grads_jx           # both real-valued DeviceArrays

    grad_L_ftorch = to_jax(grad_Lre_pt + 1j * grad_Lim_pt)
    # conjugate match
    grad_L_jax = grad_Lre_jx + 1j * grad_Lim_jx

    func_compare_complex(grad_L_ftorch, grad_L_jax, atol=atol, rtol=rtol, var_name="grad_L")


def test_discretization_forward_backward():
    """Test that discretization forward and backward passes match between PyTorch and JAX."""
    torch.manual_seed(42)
    dtype = torch.float32
    dtype_complex = torch.complex64
    
    # Test parameters
    P = 4  # Number of states
    H = 2  # Input and output feature dimensions
    atol = 1e-2
    rtol = 1e-2
    
    # Create test parameters
    Lambda = torch.complex(torch.randn(P, dtype=dtype), torch.randn(P, dtype=dtype)).to(device)
    B_tilde = torch.complex(torch.randn(P, H, dtype=dtype), torch.randn(P, H, dtype=dtype)).to(device)
    log_Delta = torch.randn(P, dtype=dtype).to(device)
    
    # Normalize for stability
    Lambda = Lambda / Lambda.abs().max()
    
    # Convert to JAX
    dtype_jax = jnp.complex64
    Lambda_jax = jnp.array(to_jax(Lambda), dtype=dtype_jax)
    B_tilde_jax = jnp.array(to_jax(B_tilde), dtype=dtype_jax)
    log_Delta_jax = jnp.array(to_jax(log_Delta))
    
    # Make parameters require gradients for PyTorch
    Lambda.requires_grad_(True)
    B_tilde.requires_grad_(True)
    log_Delta.requires_grad_(True)
    
    # Test forward pass
    Lambda_bar_torch, B_bar_torch = discretize(Lambda, B_tilde, torch.exp(log_Delta))
    Lambda_bar_jax, B_bar_jax = discretize_jax(Lambda_jax, B_tilde_jax, jnp.exp(log_Delta_jax))
    
    # Compare forward pass results
    assert jnp.allclose(to_jax(Lambda_bar_torch), Lambda_bar_jax, atol=atol, rtol=rtol), \
        "Lambda_bar forward pass does not match"
    assert jnp.allclose(to_jax(B_bar_torch), B_bar_jax, atol=atol, rtol=rtol), \
        "B_bar forward pass does not match"
    
    # Test backward pass
    # Create a simple loss function for testing gradients
    def torch_loss(Lambda, B_tilde, log_Delta):
        Lambda_bar, B_bar = discretize(Lambda, B_tilde, torch.exp(log_Delta))
        return (Lambda_bar.abs().sum() + B_bar.abs().sum()).real
    
    def jax_loss(Lambda, B_tilde, log_Delta):
        Lambda_bar, B_bar = discretize_jax(Lambda, B_tilde, jnp.exp(log_Delta))
        return (jnp.abs(Lambda_bar).sum() + jnp.abs(B_bar).sum()).real
    
    # reset gradients
    for param in [Lambda, B_tilde, log_Delta]:
        param.grad = None
    
    # PyTorch backward pass
    loss_torch = torch_loss(Lambda, B_tilde, log_Delta)
    loss_torch.backward()
    
    grad_Lambda_torch = Lambda.grad.clone().detach()
    grad_B_tilde_torch = B_tilde.grad.clone().detach()
    grad_log_Delta_torch = log_Delta.grad.clone().detach()
    
    # JAX backward pass
    grads_jax = jax.grad(jax_loss, (0, 1, 2))(Lambda_jax, B_tilde_jax, log_Delta_jax)
    grad_Lambda_jax, grad_B_tilde_jax, grad_log_Delta_jax = grads_jax
    
    # Compare gradients - handle complex conjugate issue
    torch_grad_Lambda_np = to_jax(grad_Lambda_torch)
    torch_grad_B_tilde_np = to_jax(grad_B_tilde_torch)
    torch_grad_log_Delta_np = to_jax(grad_log_Delta_torch)
    
    # Check if gradients are conjugates (common issue between PyTorch and JAX)
    func_compare_complex(torch_grad_Lambda_np, grad_Lambda_jax.conj(), atol=atol, rtol=rtol, var_name="Lambda")
    func_compare_complex(torch_grad_B_tilde_np, grad_B_tilde_jax.conj(), atol=atol, rtol=rtol, var_name="B_tilde")
    func_compare_complex(torch_grad_log_Delta_np, grad_log_Delta_jax.conj(), atol=atol, rtol=rtol, var_name="log_Delta")
    
