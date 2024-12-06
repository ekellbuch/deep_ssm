"""
Compare jax and torch S5 layer implementations.

"""
import pytest
import torch
import jax
import jax.numpy as jnp
from typing import Tuple
from deep_ssm.mixers.s5_fjax.jax_func import associative_scan

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
    Lambda_elements = jnp.repeat(Lambda_bar[None, ...], input_sequence.shape[0], axis=0)
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)
    elements = (Lambda_elements, Bu_elements) # (L, P), (L, P)

    # Compute latent state sequence given input sequence using parallel scan
    _, xs = parallel_scan(binary_operator_jax, elements) # (L, P)

    if bidirectional:
       _, xs2 = parallel_scan(binary_operator_jax, elements, reverse=True) # (L, P)
       xs = jnp.concatenate([xs, xs2], axis=0)

    # Compute SSM output sequence
    if conj_sym:
        ys = jax.vmap(lambda x, u: 2 * (C_tilde @ x).real + D * u)(xs, input_sequence)
    else:
        ys = jax.vmap(lambda x, u: (C_tilde @ x + D * u).real)(xs, input_sequence)
    return ys, xs[-1]


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


# ----------------- PyTorch functions ----------------- 
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


if load_triton:
 
    @triton.jit
    def complex_mul(a_real, a_imag, b_real, b_imag):
        """Compute complex multiplication."""
        return (a_real * b_real - a_imag * b_imag,
                a_real * b_imag + a_imag * b_real)


    @triton.jit
    def first_order_op_complex(l_real, l_imag, l_gates_real, l_gates_imag,
                            r_real, r_imag, r_gates_real, r_gates_imag):
        """Compute the first-order operation directly with real/imag components.

        A_i, Bu_i = (l_gates_real, l_real)
        A_j, Bu_j = (r_gates_real, r_real)
        return 
            A_j * A_i
            A_j * Bu_i + Bu_j
        """
        # Complex multiplication for the gate update
        # f = A_j * A_i
        f_real, f_imag = complex_mul(r_gates_real, r_gates_imag,l_gates_real, l_gates_imag)

        # Complex multiplication and addition for state update
        # mul =   A_j * Bu_i + Bu_j
        mul_real, mul_imag = complex_mul(r_gates_real, r_gates_imag, l_real, l_imag)

        mul_real = mul_real + r_real
        mul_imag = mul_imag + r_imag

        return mul_real, mul_imag, f_real, f_imag


    @triton.jit
    def forward_scan_complex(
        gates_real,
        gates_imag,
        tokens_real,
        tokens_imag,
        output_gates_real,
        output_gates_imag,
        output_tokens_real,
        output_tokens_imag,
        SEQUENCE_LENGTH: tl.constexpr,
        ):
        """
        Forward scan with direct complex number handling.
        """
        sequence_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
        strides = 1*(tl.arange(0, SEQUENCE_LENGTH) + sequence_id * SEQUENCE_LENGTH)

        # Load data
        #breakpoint()
        gates_r = tl.load(gates_real + strides)
        gates_i = tl.load(gates_imag + strides)
        tokens_r = tl.load(tokens_real + strides)
        tokens_i = tl.load(tokens_imag + strides)

        print('\n fwd triton in gates_r, gates_i, tokens_r, tokens_i:')
        print(gates_r, gates_i,tokens_r, tokens_i)

        # Perform scan operation
        tokens_new_r, tokens_new_i, gates_r_out, gates_i_out = tl.associative_scan(
            (tokens_r, tokens_i, gates_r, gates_i),
            axis=0,
            combine_fn=first_order_op_complex
        )
        print("\n fwd triton out gates_r_out, gates_i_out:, tokens_new_r, tokens_new_i, ")
        print(gates_r_out, gates_i_out,tokens_new_r, tokens_new_i)

        # Store results
        tl.store(output_gates_real + strides, gates_r_out)
        tl.store(output_gates_imag + strides, gates_i_out)
        tl.store(output_tokens_real + strides, tokens_new_r)
        tl.store(output_tokens_imag + strides, tokens_new_i)

    @triton.jit
    def backward_scan_complex(
        gates_real,
        gates_imag,
        grad_real,
        grad_imag,
        d_gates_real,
        d_gates_imag,
        d_tokens_real,
        d_tokens_imag,
        SEQUENCE_LENGTH: tl.constexpr,
        ):
        """Backward scan with direct complex number handling."""
        sequence_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
        forward_strides = (tl.arange(0, SEQUENCE_LENGTH) + sequence_id * SEQUENCE_LENGTH)
        reverse_strides = (tl.num_programs(axis=0) * tl.num_programs(axis=1) * SEQUENCE_LENGTH - 1) - forward_strides

        # Load data in reverse order
        gates_r = tl.load(gates_real + reverse_strides)
        gates_i = tl.load(gates_imag + reverse_strides)
        grad_r = tl.load(grad_real + reverse_strides)
        grad_i = tl.load(grad_imag + reverse_strides)

        print("\n bwd triton in gates_r, gates_i, grad_r, grad_i: )should be flipped)")
        print(gates_r, gates_i, grad_r, grad_i)

        #breakpoint()
        # Perform backward scan
        tokens_new_r, tokens_new_i, gates_r_out, gates_i_out = tl.associative_scan(
            (grad_r, grad_i, gates_r, gates_i),
            axis=0,
            combine_fn=first_order_op_complex
        )
        # Debug intermediate results
        print("\nbwd triton out gates_r_out, gates_i_out, tokens_new_r, tokens_new_i")
        print( gates_r_out, gates_i_out,tokens_new_r, tokens_new_i)

        #print('d_tokens_real computed', tokens_new_r)
        #print('d_gates_real computed', gates_r_out)
        # Store results in reverse order
        tl.store(d_tokens_real + reverse_strides, tokens_new_r)
        tl.store(d_tokens_imag + reverse_strides, tokens_new_i)
        tl.store(d_gates_real + reverse_strides, gates_r_out)
        tl.store(d_gates_imag + reverse_strides, gates_i_out)


class ScanBCT(torch.autograd.Function):
    
    @staticmethod
    def forward(gates, tokens):
        B, C, T = gates.shape
        assert tokens.shape == (B, C, T), "Tokens must have the same shape as gates."
        assert gates.is_contiguous(), "Gates tensor must be contiguous."
        assert tokens.is_contiguous(), "Tokens tensor must be contiguous."

        # Split into real and imaginary parts
        gates_real, gates_imag = gates.real, gates.imag
        tokens_real, tokens_imag = tokens.real, tokens.imag

        # Allocate output tensors
        gates_new = torch.zeros_like(gates, dtype=gates.dtype)
        tokens_new = torch.zeros_like(tokens, dtype=tokens.dtype)

        gates_new_real = gates_new.real
        gates_new_imag = gates_new.imag
        tokens_new_real = tokens_new.real
        tokens_new_imag = tokens_new.imag

        print("\nfwd gates_real, gates_imag,tokens_real, tokens_imag, :")
        print(gates_real, gates_imag,tokens_real, tokens_imag)
        # Forward pass
        forward_scan_complex[(B, C)](
            gates_real, gates_imag,
            tokens_real, tokens_imag,
            gates_new_real, gates_new_imag,
            tokens_new_real, tokens_new_imag,
            SEQUENCE_LENGTH=T,
            enable_fp_fusion=False
        )
        print("\n fwd from triton gates_new, tokens_new:")
        print(gates_new, tokens_new)
        return gates_new, tokens_new

    @staticmethod
    def backward(ctx, grad_gates, grad_tokens):
        # Given a linear scan: x_{t+1} = g_{t+1} * x_t + b_{t+1}
        # dL / dx_t = dL / dx_{t+1} * dx_{t+1}/dx_t = dL / dx_{t+1} * g*_{t+1}
        # grad_gates: dL / dg_t = dL / dx_{t+1} * x*_{t}
        # grad_tokens: dL / db_t = dL / dx_t
    
        # Padded reverse scan
        gates, tokens,  output_tokens = ctx.saved_tensors
        B, C, T = tokens.shape

        # Allocate output tensors
        d_tokens_real = torch.zeros_like(tokens.real, dtype=tokens.real.dtype)
        d_tokens_imag = torch.zeros_like(tokens.imag, dtype=tokens.imag.dtype)
        d_gates_real = torch.zeros_like(gates.real, dtype=gates.real.dtype)
        d_gates_imag = torch.zeros_like(gates.imag, dtype=gates.imag.dtype)

        # Split into real and imaginary parts
        gates = gates.conj()
        gates_real, gates_imag = gates.real, gates.imag
        grad_tokens_real, grad_tokens_imag = grad_tokens.real, grad_tokens.imag

        # Create shifted gates for backward pass
        padded_shifted_gates_real = torch.cat([gates_real, torch.ones_like(gates_real[..., :1])], dim=-1)[..., 1:].contiguous()
        padded_shifted_gates_imag = torch.cat([gates_imag, torch.zeros_like(gates_imag[:, :, :1])], dim=-1)[:, :, 1:].contiguous()

        print('\n bwd to triton, padded_shifted_gates_real, padded_shifted_gates_imag, grad_tokens_real, grad_tokens_imag:')
        print(padded_shifted_gates_real, padded_shifted_gates_imag, grad_tokens_real, grad_tokens_imag)
        # Backward scan
        backward_scan_complex[(B, C)](
            padded_shifted_gates_real, padded_shifted_gates_imag,
            grad_tokens_real, grad_tokens_imag,
            d_gates_real, d_gates_imag,
            d_tokens_real, d_tokens_imag,
            SEQUENCE_LENGTH=T,
            enable_fp_fusion=False
        )
        #print('d_tokens_real computed', d_tokens_real)
        #print('d_gates_real computed', d_gates_real)
        # grad_gates: dL / dg_t = dL / dx_{t+1} * x*_{t}
       
        print("bwd d_gates_real, d_gates_imag, d_tokens_real, d_tokens_imag:")
        print(d_gates_real, d_gates_imag, d_tokens_real, d_tokens_imag)
    
        padded_outputs = torch.cat([torch.zeros_like(output_tokens[..., :1]), output_tokens], dim=-1)[..., :-1]
        d_gates =  torch.complex(d_tokens_real,d_tokens_imag) * padded_outputs.conj() 

        # grad_tokens: dL / db_t = dL / dx_{t+1} * g*_{t+1}
        d_tokens = torch.complex(d_tokens_real, d_tokens_imag) #output_tokens.conj()
        
        return d_gates, d_tokens

    def vmap(info, in_dims, gates, tokens):
        """
        Vectorized map for the scan operation:
        Args:
            info: Additional information from the framework (ignored in this case).
            in_dims: A tuple specifying which dimension to map over.
            gates: Batched gates tensor (B, C, T).
            tokens: Batched tokens tensor (B, C, T).
        Returns:
            Outputs of the scan operation, vectorized across the batch dimension.
        """

        # Unpack the mapping dimensions
        gate_dim, token_dim = in_dims

        if gate_dim == token_dim:
            # Both gates and tokens are batched (B_vmap, C, T)
            return torch.vmap(ScanBCT.apply, in_dims=(0, 0))(gates, tokens)
        # Case when gates are shared (None) and mapping is over tokens
        elif gate_dim is None and token_dim == 0:
            batch_size = tokens.shape[0]
            outputs = [ScanBCT.apply(gates, tokens[i]) for i in range(batch_size)]
            #outputs = [Scan.apply(gates[None,...], x[None,...]) for x in tokens]
            output_gates, output_tokens = zip(*outputs)
            output_gates = torch.stack(output_gates, dim=0)#.squeeze(0)
            output_tokens = torch.stack(output_tokens, dim=0).squeeze(0)
            outputs =output_gates, output_tokens
            # output_gates = torch.Size([1, 1, 2]) 
            # output_tokens = torch.Size([1, 1, 2]) 
            output_dims = (0, None)
        #    return torch.vmap(lambda x: Scan.apply(gates, x))(tokens)
        ## Case when gates and tokens are both mapped (same dimensions)
        #elif gate_dim == token_dim:
        #    return torch.vmap(Scan.apply)(gates, tokens)
        else:
            raise NotImplementedError("vmap over mismatched dimensions is not supported.")

        return outputs, output_dims

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # Define the context setup required for functorch transforms
        gates, tokens = inputs
        _, output_tokens = outputs
        ctx.save_for_backward(gates, tokens, output_tokens)  # Example: saving inputs that may be needed later
        return ctx  # Return context


    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # Define the context setup required for functorch transforms
        gates, tokens = inputs
        _, output_tokens = outputs
        ctx.save_for_backward(gates, tokens, output_tokens)  # Example: saving inputs that may be needed later
        return ctx  # Return context


def scan_tri_complex(gates, tokens):
    """
    Solve a first-order recurrence relation for complex numbers.
    x_t = a_t x_{t-1} + b_t
    where :math:`a_t` ("gates") and :math:`b_t` ("tokens") are sequences of vectors.

    """
    gates = gates.contiguous()
    tokens = tokens.contiguous()
    return ScanBCT.apply(gates, tokens)



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
    #Lambda_bars = Lambda_bars.tile(input_sequence.shape[0], 1)
    Lambda_real = Lambda_bars.real.repeat(input_sequence.shape[0], 1)
    Lambda_imag = Lambda_bars.imag.repeat(input_sequence.shape[0], 1)
    Lambda_bars = torch.complex(Lambda_real, Lambda_imag)

  #Lambda_bars[0] = Lambda_bars[0] * prev_state
  # compute state sequence using associative scan: x_{t+1} = A x_t + B u
  
  # fails backward pass
  #new_gates, xs = associative_scan(binary_operator, (Lambda_bars, Bu_elements))
  
  # passes 
  #new_gates, xs = torch_associative_scan(Lambda_bars, Bu_elements)
  
  #new_gates2, xs2 = torch_associative_scan(Lambda_bars, Bu_elements)
  new_gates, xs = scan_tri_complex(Lambda_bars.mT[None,...], Bu_elements.mT[None,...])
  new_gates = new_gates.squeeze(0).mT
  xs = xs.squeeze(0).mT
  if bidirectional:
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


seqlens = [2]#, 16]#[4, 16]# [4, 32, 64]
num_statess = [1]#[4, 16, 32]
d_models = [1]#[4]


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

    atol = 1e-3
    rtol = 1e-2

    #Lambda = torch.complex(torch.randn(P, dtype=dtype), torch.randn(P, dtype=dtype)).to(device)
    #B_tilde = torch.complex(torch.randn(P, H, dtype=dtype), torch.randn(P, H, dtype=dtype)).to(device)
    #C_tilde = torch.complex(torch.randn(H, P, dtype=dtype), torch.randn(H, P, dtype=dtype)).to(device)
    Lambda = torch.complex(torch.randn(P, dtype=dtype), torch.zeros(P, dtype=dtype)).to(device)
    B_tilde = torch.complex(torch.randn(P, H, dtype=dtype), torch.zeros(P, H, dtype=dtype)).to(device)
    C_tilde = torch.complex(torch.randn(H, P, dtype=dtype), torch.zeros(H, P, dtype=dtype)).to(device)
    D = torch.randn(H, dtype=dtype).to(device)
    log_Delta = torch.randn(P, dtype=dtype).to(device)

    params = (Lambda, B_tilde, C_tilde, D, log_Delta)
    params = [param.requires_grad_(True) for param in params]
    input_sequences = torch.randn(B, L, H, dtype=dtype).to(device)


    # Now let's compare the jax and torch versions
    dtype_jax= jnp.complex64
    Lambda_jax = jnp.array(Lambda.cpu().detach().numpy(), dtype=dtype_jax)
    B_tilde_jax = jnp.array(B_tilde.cpu().detach().numpy(), dtype=dtype_jax)
    C_tilde_jax = jnp.array(C_tilde.cpu().detach().numpy(), dtype=dtype_jax)
    D_jax = jnp.array(D.cpu().detach().numpy())
    log_Delta_jax = jnp.array(log_Delta.cpu().detach().numpy())

    assert jnp.allclose(Lambda.cpu().detach().numpy(), Lambda_jax, atol=1e-6)
    assert jnp.allclose(B_tilde.cpu().detach().numpy(), B_tilde_jax, atol=1e-6)
    assert jnp.allclose(C_tilde.cpu().detach().numpy(), C_tilde_jax, atol=1e-6)
    assert jnp.allclose(D.cpu().detach().numpy(), D_jax, atol=1e-6)
    assert jnp.allclose(log_Delta.cpu().detach().numpy(), log_Delta_jax, atol=1e-6)

    params_jax = (Lambda_jax, B_tilde_jax, C_tilde_jax, D_jax, log_Delta_jax)
    input_sequences_jax = jnp.array(input_sequences.cpu().detach().numpy())

    # compare discretization
    Lambda_bar_torch, B_bar_torch = discretize(Lambda, B_tilde, log_Delta)
    Lambda_bar_jax, B_bar_jax = discretize_jax(Lambda_jax, B_tilde_jax, log_Delta_jax)

    assert jnp.allclose(Lambda_bar_torch.cpu().detach().numpy(), Lambda_bar_jax, atol=atol), print("Failed discretization")
    assert jnp.allclose(B_bar_torch.cpu().detach().numpy(), B_bar_jax, atol=atol), print("Failed discretization")

    # compare apply ssm
    torch_output, torch2 = apply_ssm(Lambda_bar_torch, B_bar_torch, C_tilde, D, input_sequences[0])
    jax_output, jaz2 = apply_ssm_jax(Lambda_bar_jax, B_bar_jax, C_tilde_jax, D_jax, input_sequences_jax[0])
    
    breakpoint()
    assert jnp.allclose(torch_output.cpu().detach().numpy(), jax_output, atol=atol, rtol=rtol), print(f"apply_ssm out1 torch{torch_output.cpu().detach().numpy()} jax {jax_output}")
    assert jnp.allclose(torch2.cpu().detach().numpy(), jaz2, atol=atol, rtol=rtol), print(f"apply_ssm out2 do not match torch{torch2.cpu().detach().numpy()} jax {jaz2}")

    # Single sequence example
    output_sequence = batch_apply_S5_layer(params, input_sequences)

    #print("Single sequence output:")
    assert output_sequence.shape == (B, L, H)

    output_sequence_jax = batch_apply_S5_layer_jax(params_jax, input_sequences_jax)
    assert output_sequence.shape == output_sequence_jax.shape
    assert jnp.allclose(output_sequence.cpu().detach().numpy(), output_sequence_jax, atol=atol, rtol=rtol), "Outputs of S5 layer do not match"

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
    for param_id, (torch_grad, jax_grad) in enumerate(zip(torch_grads, jax_grads)):
        # Convert PyTorch gradients to numpy
        torch_grad_np = torch_grad.cpu().detach().numpy()
        if jnp.iscomplexobj(jax_grad) or jnp.iscomplexobj(torch_grad_np):
            # pytorch uses the conjugate transpose for complex gradients: (https://github.com/jax-ml/jax/issues/9110)
            # If complex, compare real and imaginary parts separately
            assert jnp.allclose(torch_grad_np.real, jax_grad.real, atol=atol, rtol=rtol), \
                f"Real parts of gradients {param_id} do not match: {torch_grad_np.real} vs {jax_grad.real}"
            assert jnp.allclose(-1*torch_grad_np.imag, jax_grad.imag, atol=atol, rtol=rtol), \
                f"Imaginary parts of gradients {param_id} do not match: {-1*torch_grad_np.imag} vs {jax_grad.imag}"
        else:
            # If real, compare directly
            assert jnp.allclose(torch_grad_np, jax_grad, atol=atol, rtol=rtol), \
                f"Gradients {param_id} do not match: {torch_grad_np} vs {jax_grad}"

