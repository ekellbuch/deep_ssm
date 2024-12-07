import torch
from torch.autograd import Function
import triton
import triton.language as tl
import os
import pytest

os.environ["TRITON_INTERPRET"]="1"
os.environ["TRITON_LOG_LEVEL"] = "debug"


def torch_associative_scanBTC(gates, tokens):
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


def binary_operator(
  q_i, q_j
):
  """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
  Args:
      q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
      q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
  Returns:
      new element ( A_out, Bu_out )
      A_out = A_j * A_i
      Bu_out = A_j * b_i + b_j
  """
  A_i, b_i = q_i
  A_j, b_j = q_j

  return A_j * A_i,  torch.addcmul(b_j, A_j, b_i)



class ComplexLinearScanBTC(Function):
    @staticmethod
    def forward(ctx, gates, tokens):
        """
        Forward pass for the complex linear scan.
        x_t = g_t * x_{t-1} + b_t
        Args:
            gates: Complex tensor of shape (batch, len, dim).
            tokens: Complex tensor of shape (batch, len, dim).
        Returns:
            outputs: Complex tensor of shape (batch, len, dim).
        """
        seq_len = tokens.shape[-2]
        outputs = torch.zeros_like(tokens, dtype=torch.cfloat)

        # Initialize (a_prefix, b_prefix) variables
        a_prefix = gates[...,0, :]
        b_prefix = tokens[..., 0, :]

        # initialize x_0 = a_0 * x_0 + b_prefix,  but we assume x_0 = 0
        outputs[..., 0, :] = tokens[..., 0, :]

        for t in range(1, seq_len):
            #outputs[..., t] = gates[..., t] * outputs[..., t-1] + tokens[..., t]

            # (a_prefix, b_prefix) = (a_prefix, b_prefix) o (gates[..., t], tokens[..., t])
            a_prefix, b_prefix = binary_operator((a_prefix, b_prefix), (gates[..., t, :], tokens[..., t, :]))
            # x_t = a_prefix * x_0 + b_prefix
            outputs[..., t, :] = b_prefix
        ctx.save_for_backward(gates, tokens, outputs)
        return outputs

        
    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Backward pass for the complex linear scan.
        Args:
            grad_outputs: Complex tensor of shape (batch, dim, len).
        Returns:
            Gradients for tokens, gates, and contributions (all complex tensors).
        """
        gates, tokens, outputs = ctx.saved_tensors
        seq_len = tokens.shape[-2]

        # Initialize gradients
        grad_tokens = torch.zeros_like(tokens, dtype=tokens.dtype)
        grad_gates = torch.zeros_like(gates, dtype=gates.dtype)

        # Gradient of the loss with respect to the last output
        padded_shifted_gates = torch.cat([gates, torch.ones_like(gates[..., :1, :])], dim=-2)[..., 1:, :].contiguous().conj()

        padded_shifted_gates_rev = torch.flip(padded_shifted_gates, dims=[-2])
        grad_outputs_rev = torch.flip(grad_outputs, dims=[-2])

        # Forward scan
        grad_tokens[..., 0, :] = grad_outputs_rev[..., 0, :]
        a_prefix = padded_shifted_gates_rev[..., 0, :]
        b_prefix = grad_outputs_rev[..., 0, :]

        for t in range(1, seq_len):
            # (a_prefix, b_prefix) = (a_prefix, b_prefix) o (shifted_gates[..., t], grads[..., t])
            a_prefix, b_prefix = binary_operator((a_prefix, b_prefix), (padded_shifted_gates_rev[..., t, :], grad_outputs_rev[..., t, :]))
            grad_tokens[..., t, :] = b_prefix

        padded_outputs = torch.cat([torch.zeros_like(outputs[..., :1, :]), outputs], dim=-2)[..., :-1, :]

        grad_tokens = torch.flip(grad_tokens, dims=[-2])
        grad_gates = grad_tokens * padded_outputs.conj()
        return grad_gates, grad_tokens

# -------------------------------------------------------------------------------------------------
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
    f_real, f_imag = complex_mul(r_gates_real, r_gates_imag, l_gates_real, l_gates_imag)

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
    CHANNELS: tl.constexpr,
    ):
    """
    Forward scan with direct complex number handling.
    """
    # Program IDs for Batch and Channel (dimensions along which we parallelize)
    batch_id = tl.program_id(axis=0)  # Batch index
    channel_id = tl.program_id(axis=1)  # Channel index

    # When we want to move between batches, we need to stride T*C elements
    # Within a given batch, we need to stride C elements

    # Compute base offset for the batch and channel
    base_offset = batch_id * SEQUENCE_LENGTH * CHANNELS + channel_id

    # Sequence strides: Stride along the sequence dimension (T)
    strides = base_offset + tl.arange(0, SEQUENCE_LENGTH) * CHANNELS

    # Remember we are dealing with complex numbers which are interleaved in memory
    strides = strides * 2

    # Return a tensor whose values are loaded from memory at located specified by the pointer
    tokens_r = tl.load(tokens_real + strides)
    tokens_i = tl.load(tokens_imag + strides)
    gates_r = tl.load(gates_real + strides)
    gates_i = tl.load(gates_imag + strides)

    #print("fwd in triton: gates_r, gates_i, tokens_r, tokens_i:")
    #print(gates_r, gates_i, tokens_r, tokens_i)
    # Perform scan operation
    tokens_new_r, tokens_new_i, gates_r_out, gates_i_out = tl.associative_scan(
        (tokens_r, tokens_i, gates_r, gates_i),
        axis=0,
        combine_fn=first_order_op_complex
    )

    # print("fwd out triton: gates_r_out, gates_i_out, tokens_new_r, tokens_new_i:")
    # print(gates_r_out, gates_i_out, tokens_new_r, tokens_new_i)

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
    CHANNELS: tl.constexpr,
    ):
    """Backward scan with direct complex number handling."""

    # Program IDs for Batch and Channel (dimensions along which we parallelize)
    batch_id = tl.program_id(axis=0)  # Batch index
    channel_id = tl.program_id(axis=1)  # Channel index

    # Same as forward scan but read sequence in reverse order
    base_offset = batch_id * SEQUENCE_LENGTH * CHANNELS + channel_id
    strides = base_offset + tl.arange(0, SEQUENCE_LENGTH) * CHANNELS
    reversed_strides = base_offset + (SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH)) * CHANNELS

    # Remember we are dealing with complex numbers which are interleaved in memory
    # reversed_strides = reversed_strides * 2
    # However, 

    # Load data in reverse order
    gates_r = tl.load(gates_real + reversed_strides)
    gates_i = tl.load(gates_imag + reversed_strides)
    grad_r = tl.load(grad_real + reversed_strides * 2)
    grad_i = tl.load(grad_imag + reversed_strides * 2)

    #print("bwd in triton:gates_r, gates_i, grad_r, grad_i:")
    #print(gates_r, gates_i, grad_r, grad_i)
    
    # Perform backward scan
    tokens_new_r, tokens_new_i, gates_r_out, gates_i_out = tl.associative_scan(
        (grad_r, grad_i, gates_r, gates_i),
        axis=0,
        combine_fn=first_order_op_complex
    )
    # Store results in reverse order
    #print("bwd out triton:gates_r_out, gates_i_out, tokens_new_r, tokens_new_i:")
    #print(gates_r_out, gates_i_out, tokens_new_r,tokens_new_i)
    
    # when we store we need to reverse the strides
    tl.store(d_tokens_real + reversed_strides, tokens_new_r)
    tl.store(d_tokens_imag + reversed_strides, tokens_new_i)
    tl.store(d_gates_real + strides, gates_r_out)
    tl.store(d_gates_imag + strides, gates_i_out)


class ScanBTC(torch.autograd.Function):
    
    @staticmethod
    def forward(gates, tokens):
        B, T, C = gates.shape
        assert tokens.shape == (B, T, C), "Tokens must have the same shape as gates."
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

        #print("fwd: gates_real, gates_imag,tokens_real, tokens_imag, :")
        #print(gates_real, gates_imag,tokens_real, tokens_imag)

        # Forward pass
        forward_scan_complex[(B, C)](
            gates_real, gates_imag,
            tokens_real, tokens_imag,
            gates_new_real, gates_new_imag,
            tokens_new_real, tokens_new_imag,
            SEQUENCE_LENGTH=T,
            CHANNELS=C,
            enable_fp_fusion=False
        )

        #print("fwd out: gates_new, tokens_new:")
        #print(gates_new, tokens_new)

        return gates_new, tokens_new

    @staticmethod
    def backward(ctx, grad_gates, grad_tokens):
        # Given a linear scan: x_{t+1} = g_{t+1} * x_t + b_{t+1}
        # dL / dx_t = dL / dx_{t+1} * dx_{t+1}/dx_t = dL / dx_{t+1} * g*_{t+1}
        # grad_gates: dL / dg_t = dL / dx_{t+1} * x*_{t}
        # grad_tokens: dL / db_t = dL / dx_t
    
        # Padded reverse scan
        gates, tokens,  output_tokens = ctx.saved_tensors
        B, T, C = tokens.shape

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
        padded_shifted_gates_real = torch.cat([gates_real, torch.ones_like(gates_real[..., :1, :])], dim=-2)[..., 1:, :].contiguous()
        padded_shifted_gates_imag = torch.cat([gates_imag, torch.zeros_like(gates_imag[..., :1, :])], dim=-2)[..., 1:, :].contiguous()

        #print("bwd: padded_shifted_gates_real, padded_shifted_gates_imag, grad_tokens_real, grad_tokens_imag:")
        #print(padded_shifted_gates_real.mT, padded_shifted_gates_imag.mT, grad_tokens_real.mT, grad_tokens_imag.mT)
        # Backward scan
        backward_scan_complex[(B, C)](
            padded_shifted_gates_real, padded_shifted_gates_imag,
            grad_tokens_real, grad_tokens_imag,
            d_gates_real, d_gates_imag,
            d_tokens_real, d_tokens_imag,
            SEQUENCE_LENGTH=T,
            CHANNELS=C,
            enable_fp_fusion=False
        )
        #print('d_gates_real computed', d_gates_real.mT, d_gates_imag.mT)
        #print('d_tokens_real computed', d_tokens_real.mT, d_tokens_imag.mT)

        # grad_gates: dL / dg_t = dL / dx_{t+1} * x*_{t}       
        padded_outputs = torch.cat([torch.zeros_like(output_tokens[..., :1, :]), output_tokens], dim=-2)[..., :-1, :]
        d_gates =  torch.complex(d_tokens_real,d_tokens_imag) * padded_outputs.conj()

        # grad_tokens: dL / db_t = dL / dx_{t+1} * g*_{t+1}
        d_tokens = torch.complex(d_tokens_real, d_tokens_imag) # * gates.conj()

        return d_gates, d_tokens

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # Define the context setup required for functorch transforms
        gates, tokens = inputs
        _, output_tokens = outputs
        ctx.save_for_backward(gates, tokens, output_tokens)  # Example: saving inputs that may be needed later
        return ctx  # Return context
        
def scan_BTC_complex(gates, tokens):
    """
    Solve a first-order recurrence relation for complex numbers.
    x_t = a_t x_{t-1} + b_t
    where :math:`a_t` ("gates") and :math:`b_t` ("tokens") are sequences of vectors.

    """
    gates = gates.contiguous()
    tokens = tokens.contiguous()
    return ScanBTC.apply(gates, tokens)



only_reals =[True, False]
complexities = ["v0", "v1", "v2", "v3", "v4"]
batch_dims = [1,2,4, 8]
feature_dims = [1, 2, 4]
seq_lens = [4, 8, 16, 256]

"""
only_reals =[False]
complexities = ["v3"]
batch_dims = [2]
feature_dims = [1]
seq_lens = [4]
"""

def create_inputs(batch, dim, seq_len, complexity, only_real):
    if complexity == "v0":
        gates = torch.ones(batch, seq_len, dim, dtype=torch.cfloat)
        tokens = torch.arange(batch*dim*seq_len).view(batch, seq_len, dim).type(torch.cfloat)/(batch*dim*seq_len)
    elif complexity == "v1":
        gates = torch.arange(batch*dim*seq_len).view(batch, seq_len, dim).type(torch.cfloat)/(batch*dim*seq_len)
        tokens = torch.arange(batch*dim*seq_len).view(batch,seq_len, dim).type(torch.cfloat)/(batch*dim*seq_len)
    elif complexity == "v2":
        gates = torch.ones(batch, seq_len, dim, dtype=torch.cfloat)
        tokens = torch.randn(batch, seq_len, dim, dtype=torch.cfloat)
    elif complexity == "v3":
        gates = torch.randn(batch*dim*seq_len).view(batch, seq_len, dim).type(torch.cfloat)
        tokens = torch.ones(batch*dim*seq_len).view(batch, seq_len, dim).type(torch.cfloat)
    elif complexity == "v4":
        gates = torch.randn(batch, seq_len, dim, dtype=torch.cfloat)
        tokens = torch.randn(batch, seq_len, dim, dtype=torch.cfloat)

    if only_real:
        gates.imag = 0 
        tokens.imag = 0

    gates = gates.contiguous()
    tokens = tokens.contiguous()
    tokens.requires_grad_(True)
    gates.requires_grad_(True)
    return gates, tokens

@pytest.mark.parametrize("only_real", only_reals)
@pytest.mark.parametrize("complexity", complexities)
@pytest.mark.parametrize("batch", batch_dims)
@pytest.mark.parametrize("dim", feature_dims)
@pytest.mark.parametrize("seq_len", seq_lens)
def test_scan_functions(only_real, complexity, batch, dim, seq_len):
    # For a linear scan: x_{t+1} = g_{t+1} * x_t + b_{t+1}
    # compare a vanilla pytorch implementation, a custom autograd implementation, and
    torch.manual_seed(42)

    gates, tokens = create_inputs(batch, dim, seq_len, complexity, only_real)
    # ----------------------------
    # PyTorch Sequential Implementation
    tokens.grad = None  # Reset gradients
    gates.grad = None
    out_g, out_token = torch_associative_scanBTC(gates, tokens)

    # Compute a loss
    loss_seq = out_token.abs().sum()
    loss_seq.backward()

    seq_gates_grad = gates.grad.clone()
    seq_tokens_grad = tokens.grad.clone()

    # ----------------------------
    # CustomAutograd Implementation
    tokens.grad = None  # Reset gradients
    gates.grad = None

    outputs = ComplexLinearScanBTC.apply(gates,tokens)

    # Compute a loss
    loss = outputs.abs().sum()
    loss.backward()

    seq2_gates_grad = gates.grad.clone()
    seq2_tokens_grad = tokens.grad.clone()

    # ----------------------------
    # Triton Implementation
    tokens.grad = None  # Reset gradients
    gates.grad = None

    new_g, new_t = scan_BTC_complex(gates, tokens)

    # Compute a loss
    loss_tri = new_t.abs().sum()
    loss_tri.backward()

    tri_gates_grad = gates.grad.clone()
    tri_tokens_grad = tokens.grad.clone()

    # Check forward pass match
    assert torch.allclose(out_token, outputs, atol=1e-5), print(f"Forward pass mismatch vanilla {out_token} and custom {outputs}" )
    assert torch.allclose(out_token, new_t, atol=1e-5),  print(f"Forward pass mismatch vanilla {out_token} and triton {new_t}" )

    # Check backward pass match
    assert torch.allclose(seq_tokens_grad, seq2_tokens_grad, atol=1e-5), print(f"Backward pass mismatch token vanilla {seq_tokens_grad} and custom {seq2_tokens_grad}" )
    assert torch.allclose(seq_tokens_grad, tri_tokens_grad, atol=1e-5), print(f"Backward pass mismatch token vanilla {seq_tokens_grad} and triton {tri_tokens_grad}" )
    assert torch.allclose(seq_gates_grad, seq2_gates_grad, atol=1e-5), print(f"Backward pass mismatch gates vanilla {seq_gates_grad} and custom {seq2_gates_grad}" )
    assert torch.allclose(seq_gates_grad, tri_gates_grad, atol=1e-5), print(f"Backward pass mismatch gates  vanilla {seq_gates_grad} and triton {tri_gates_grad}" )


