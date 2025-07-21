"""
torch_associative_scanBCT
torch_associative_scanBTC
scan_bct_complex
scan_btc_complex
scan_btc_complex_autograd
scan_bct_complex_autograd
"""
import torch
import os
from typing import Tuple, List
try:
    import triton
    import triton.language as tl
    load_triton = True
except:
    load_triton = False
import torch.nn as nn


#os.environ["TRITON_INTERPRET"]="1"
#os.environ["TRITON_LOG_LEVEL"] = "debug"


def torch_associative_scanBCT(gates, tokens, reverse=False):
    """
    PyTorch equivalent of a parallel scan for linear recurrence using gates and tokens:

        x_t = gates_t * x_{t-1} + tokens_t

        x_0 = tokens_0

    when reverse is True, we compute the scan in reverse order:

        x_t = gates_t * x_{t+1} + tokens_t


    Args:
        gates: Complex tensor representing the gates of shape (batch_size, dim, seq_len).
        tokens: Complex tensor representing the tokens of shape (batch_size, dim,seq_len).

    Returns:
        A tuple of tensors (scanned_gates, scanned_tokens) where:
            scanned_gates: The cumulative product of gates.
            scanned_tokens: The cumulative result of applying the recurrence relation.

    """
    seq_len = gates.shape[-1]

    # Initialize outputs
    scanned_gates = torch.zeros_like(gates)
    scanned_tokens = torch.zeros_like(tokens)

    if reverse:
        # Flip the tensors along the sequence dimension
        gates = torch.flip(gates, dims=[-1])
        tokens = torch.flip(tokens, dims=[-1])

    # Set the initial values
    scanned_gates[..., 0] = gates[..., 0]
    scanned_tokens[..., 0] = tokens[..., 0]

    # Temporary tensors for computation
    current_gates = gates[..., 0]
    current_tokens = tokens[..., 0]

    for t in range(1, seq_len):
        # Compute new values
        current_gates = gates[..., t] * current_gates
        current_tokens = gates[..., t] * current_tokens + tokens[..., t]

        # Assign to output tensors
        scanned_gates[..., t] = current_gates
        scanned_tokens[..., t] = current_tokens

    if reverse:
        scanned_gates = scanned_gates.flip(dims=[-1])
        scanned_tokens = scanned_tokens.flip(dims=[-1])

    return scanned_gates, scanned_tokens


def torch_associative_scanBTC(gates, tokens, reverse=False):
    """
    PyTorch equivalent of a parallel scan for linear recurrence using gates and tokens.
    x_t = g_t * x_{t-1} + b_t

    where g_t are the gates and b_t are the tokens.

    Args:
        gates: Complex tensor representing the gates of shape (batch_size, seq_len, dim).
        tokens: Complex tensor representing the tokens of shape (batch_size, seq_len, dim).

    Returns:
        A tuple of tensors (scanned_gates, scanned_tokens) where:
            scanned_gates: The cumulative product of gates.
            scanned_tokens: The cumulative result of applying the recurrence relation.

    """
    seq_len = gates.shape[-2]

    if reverse:
        gates = torch.flip(gates, dims=[-2])
        tokens = torch.flip(tokens, dims=[-2])

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

    if reverse:
        scanned_gates = scanned_gates.flip(dims=[-2])
        scanned_tokens = scanned_tokens.flip(dims=[-2])

    return scanned_gates, scanned_tokens



@triton.jit
def complex_mul(a_real, a_imag, b_real, b_imag):
    return a_real * b_real - a_imag * b_imag, a_real * b_imag + a_imag * b_real


@triton.jit
def forward_scan_bct_complex(
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
    REVERSE: tl.constexpr,
    ):
    """
    Forward scan with direct complex number handling.
    """
    # Program IDs for Batch and Channel (dimensions along which we parallelize)
    batch_id = tl.program_id(axis=0)  # Batch index
    channel_id = tl.program_id(axis=1)  # Channel index

    # When we want to move between batches, we need to skip T*C elements
    # Moving between channels within a batch requires skipping T elements

    # Base offset for batch and channel in the BCT layout
    base_offset = batch_id * CHANNELS * SEQUENCE_LENGTH + channel_id * SEQUENCE_LENGTH

    # Sequence strides: Stride along the sequence dimension (T)
    if REVERSE:
        seq_idx = SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH)
    else:
        seq_idx = tl.arange(0, SEQUENCE_LENGTH)

    strides = base_offset + seq_idx

    # we are dealing with complex numbers with real and imag parts interleaved in memory
    strides = strides * 2

    # Return a tensor whose values are loaded from memory at located specified by the pointer
    tokens_r = tl.load(tokens_real + strides)
    tokens_i = tl.load(tokens_imag + strides)
    gates_r = tl.load(gates_real + strides)
    gates_i = tl.load(gates_imag + strides)

    # Scan outputs: cummulative coefficience, final state
    tokens_new_r, tokens_new_i, gates_r_out, gates_i_out = tl.associative_scan(
        (tokens_r, tokens_i, gates_r, gates_i),
        axis=0,
        combine_fn=first_order_op_complex,
        reverse=False
    )

    out_idx = base_offset + seq_idx * CHANNELS
    out_idx2 = out_idx * 2


    # Store results
    tl.store(output_gates_real + strides, gates_r_out)
    tl.store(output_gates_imag + strides, gates_i_out)
    tl.store(output_tokens_real + strides, tokens_new_r)
    tl.store(output_tokens_imag + strides, tokens_new_i)

@triton.jit
def backward_scan_bct_complex(
    gates_real,
    gates_imag,
    grad_real,
    grad_imag,
    d_gates_real,
    d_gates_imag,
    d_tokens_real,
    d_tokens_imag,
    SEQUENCE_LENGTH: tl.constexpr,
    REVERSE: tl.constexpr,
    CHANNELS: tl.constexpr,
    ):
    """Backward scan with direct complex number handling."""
    # Program IDs for Batch and Channel (dimensions along which we parallelize)
    batch_id = tl.program_id(axis=0)  # Batch index
    channel_id = tl.program_id(axis=1)  # Channel index

    base_offset = batch_id * CHANNELS * SEQUENCE_LENGTH + channel_id * SEQUENCE_LENGTH

    # For backward scan, read in reverse order
    if REVERSE:
        seq_idx = tl.arange(0, SEQUENCE_LENGTH)
    else:
        seq_idx = (SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH))
    
    reversed_strides = base_offset + seq_idx
    
    # Load data in reverse order for backward scan
    gates_r = tl.load(gates_real + reversed_strides)
    gates_i = tl.load(gates_imag + reversed_strides)
    grad_r = tl.load(grad_real + reversed_strides * 2)
    grad_i = tl.load(grad_imag + reversed_strides * 2)

    #print("\n")
    #print("gates_r, gates_i", gates_r, gates_i)
    #print("grad_r", "grad_i", grad_r, grad_i)
    
    # Perform backward scan
    tokens_new_r, tokens_new_i, gates_r_out, gates_i_out = tl.associative_scan(
        (grad_r, grad_i, gates_r, gates_i),
        axis=0,
        combine_fn=first_order_op_complex,
        reverse=False,
    )

    #print("REVERSE", REVERSE)
    #print("tokens_new_r, tokens_new_i, gates_r_out, gates_i_out:")
    #print(tokens_new_r, tokens_new_i, gates_r_out, gates_i_out)
    if REVERSE:
        rev_idx = tl.arange(0, SEQUENCE_LENGTH) 
    else:
        rev_idx = (SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH)) 
    store_idx = base_offset + rev_idx 
    tl.store(d_tokens_real + store_idx, tokens_new_r)
    tl.store(d_tokens_imag + store_idx, tokens_new_i)
    tl.store(d_gates_real + store_idx, gates_r_out)
    tl.store(d_gates_imag + store_idx, gates_i_out)


class ScanBCT(torch.autograd.Function):
    
    @staticmethod
    def forward(gates, tokens, reverse=False):
        B, C, T = gates.shape
        assert tokens.shape == (B, C, T), "Tokens must have the same shape as gates."
        assert gates.is_contiguous(), "Gates tensor must be contiguous."
        assert tokens.is_contiguous(), "Tokens tensor must be contiguous."

        # Split into real and imaginary parts
        gates_real = gates.real
        gates_imag = gates.imag
        tokens_real = tokens.real
        tokens_imag = tokens.imag

        # Allocate output tensors
        gates_new = torch.empty_like(gates, dtype=gates.dtype)
        tokens_new = torch.empty_like(tokens, dtype=tokens.dtype)

        gates_new_real = gates_new.real
        gates_new_imag = gates_new.imag
        tokens_new_real = tokens_new.real
        tokens_new_imag = tokens_new.imag

        #print("fwd gates_real, gates_imag,tokens_real, tokens_imag, :")
        #print(gates_real, gates_imag,tokens_real, tokens_imag)

        # Forward pass
        forward_scan_bct_complex[(B, C)](
            gates_real, gates_imag,
            tokens_real, tokens_imag,
            gates_new_real, gates_new_imag,
            tokens_new_real, tokens_new_imag,
            SEQUENCE_LENGTH=T,
            CHANNELS=C,
            REVERSE=reverse,
        )
        return gates_new, tokens_new

    @staticmethod
    def backward(ctx, grad_gates, grad_tokens):
        """
        Given the linear scan: x_{t+1} = g_{t+1} * x_t + b_{t+1}
        we want to compute the gradients of the loss with respect to the gates and tokens.
        """
        # Given a linear scan: x_{t+1} = g_{t+1} * x_t + b_{t+1}
        # dL / dx_t = dL / dx_{t+1} * dx_{t+1}/dx_t = dL / dx_{t+1} * g*_{t+1}
        # grad_gates: dL / dg_t = dL / dx_{t+1} * x*_{t}
        # grad_tokens: dL / db_t = dL / dx_t
    
        reverse = ctx.reverse
        # Padded reverse scan:
        gates, tokens, output_tokens = ctx.saved_tensors
        B, C, T = tokens.shape

        # Allocate output tensors
        d_tokens_real = torch.empty_like(tokens.real, dtype=tokens.real.dtype)
        d_tokens_imag = torch.empty_like(tokens.imag, dtype=tokens.imag.dtype)
        d_gates_real = torch.empty_like(gates.real, dtype=gates.real.dtype)
        d_gates_imag = torch.empty_like(gates.imag, dtype=gates.imag.dtype)

        # Split into real and imaginary parts
        gates_real = gates.real
        gates_imag = - gates.imag
        grad_tokens_real = grad_tokens.real
        grad_tokens_imag = grad_tokens.imag

        # Create shifted gates for backward pass 
        padded_shifted_gates_real = torch.empty_like(gates_real)
        padded_shifted_gates_imag = torch.empty_like(gates_imag)
            
        if reverse:
            # Fill first position with ones/zeros
            padded_shifted_gates_real[..., 0] = 1.0
            padded_shifted_gates_imag[..., 0] = 0.0
            # Copy shifted gates
            padded_shifted_gates_real[..., 1:] = gates_real[..., :-1]
            padded_shifted_gates_imag[..., 1:] = gates_imag[..., :-1]
        else:
            # Copy shifted gates
            padded_shifted_gates_real[..., :-1] = gates_real[..., 1:]
            padded_shifted_gates_imag[..., :-1] = gates_imag[..., 1:]
            # Fill last position with ones/zeros
            padded_shifted_gates_real[..., -1] = 1.0
            padded_shifted_gates_imag[..., -1] = 0.0
        
        # Backward scan
        #print('padded_shifted_gates', padded_shifted_gates_real, padded_shifted_gates_imag)
        #print('grad_tokens', grad_tokens_real, grad_tokens_imag)
        backward_scan_bct_complex[(B, C)](
            padded_shifted_gates_real, padded_shifted_gates_imag,
            grad_tokens_real, grad_tokens_imag,
            d_gates_real, d_gates_imag,
            d_tokens_real, d_tokens_imag,
            SEQUENCE_LENGTH=T,
            CHANNELS=C,
            REVERSE=reverse,
        )
        #print('d_tokens_real computed', d_tokens_real)
        #print('d_tokens_imag computed', d_tokens_imag)
        #print('d_gates_real computed', d_gates_real)
        #print('d_gates_imag computed', d_gates_imag)

        # grad_gates: dL / dg_t = dL / dx_{t+1} * x*_{t}
        # grad_tokens: dL / db_t = dL / dx_{t+1} * g*_{t+1}
        d_tokens = torch.complex(d_tokens_real, d_tokens_imag)
                
        # Compute d_gates using padded outputs 
        if reverse:
            # For x_t = a_t x_{t+1} + b_t → dL/da_t = dL/x_t * x_{t+1}
            padded_outputs = torch.cat(
                [output_tokens[..., 1:], torch.zeros_like(output_tokens[..., :1])], dim=-1
            )
            d_gates = d_tokens * padded_outputs.conj()
        else:
            # For x_t = a_t x_{t-1} + b_t → ∂L/∂a_t = ∂L/∂x_t * x_{t-1}
            padded_outputs = torch.cat(
                [torch.zeros_like(output_tokens[..., :1]), output_tokens], dim=-1
            )[..., :-1]
            d_gates = d_tokens * padded_outputs.conj()

        #print("out_grad_tokens", d_tokens)
        return d_gates, d_tokens, None

    @staticmethod
    def vmap(info, in_dims, gates, tokens, reverse=False):
        """
        A "vmap rule" is the logic of how to perform the operation given inputs with one additional dimension.        
        
        Args:
             info: Metadata or additional info for vmap (not used here).
            in_dims: Tuple specifying the dimension to slice for each input.
            gates: Input gates tensor of shape (B, T, C) if in_dims[0] is None.
            tokens: Input tokens tensor of shape (B', B, T, C) if in_dims[1] is 0.
        Returns:
            outputs: Output tensor of shape (B', B, T, C).
        """

        # Handle case where gates is constant (None in in_dims)
        if in_dims == (None, 0, None):
            # gates: (C, T) - constant across batch
            # tokens: (B, C, T) - vectorized over batch dimension
            B = tokens.shape[0]
            # Expand gates to match batch dimension
            gates_expanded = gates.unsqueeze(0).expand(B, -1, -1).contiguous()
            outputs_g, outputs_k = ScanBCT.apply(gates_expanded, tokens, reverse)
            out_dims = (0, 0)  # Both outputs vectorized over batch dimension
        else:
            raise NotImplementedError("Only (None, 0) is implemented")

        return (outputs_g, outputs_k), out_dims

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # Define the context setup required for functorch transforms
        gates, tokens, reverse = inputs
        _, output_tokens = outputs
        ctx.save_for_backward(gates, tokens, output_tokens)  # Example: saving inputs that may be needed later
        ctx.reverse = reverse
        
@torch.compiler.disable(recursive=False)
def scan_bct_complex(gates, tokens, reverse=False):
    """
    Solve a first-order recurrence relation for complex numbers.
    x_t = a_t x_{t-1} + b_t
    where :math:`a_t` ("gates") and :math:`b_t` ("tokens") are sequences of vectors.

    """
    gates = gates.contiguous()
    tokens = tokens.contiguous()
    return ScanBCT.apply(gates, tokens, reverse)


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


class ScanBTC(torch.autograd.Function):
    reverse = False  # default

    @staticmethod
    def forward(gates: torch.Tensor, tokens: torch.Tensor):

        B, T, C = gates.shape
        assert tokens.shape == (B, T, C), "Tokens must have the same shape as gates."
        assert gates.is_contiguous(), "Gates tensor must be contiguous."
        assert tokens.is_contiguous(), "Tokens tensor must be contiguous."

        # Split into real and imaginary parts
        gates_real = gates.real
        gates_imag = gates.imag
        tokens_real = tokens.real
        tokens_imag = tokens.imag

        # Allocate output tensors
        gates_new = torch.empty_like(gates, dtype=gates.dtype)
        tokens_new = torch.empty_like(tokens, dtype=tokens.dtype)

        gates_new_real = gates_new.real
        gates_new_imag = gates_new.imag
        tokens_new_real = tokens_new.real
        tokens_new_imag = tokens_new.imag

        #print("fwd: gates_real, gates_imag,tokens_real, tokens_imag, :")
        #print(gates_real, gates_imag,tokens_real, tokens_imag)
        # Forward pass
        #print(gates_real.data_ptr(), gates_imag.data_ptr(), tokens_real.data_ptr())
        reverse= ScanBTC.reverse

        forward_scan_btc_complex[(B, C)](
            gates_real, gates_imag,
            tokens_real, tokens_imag,
            gates_new_real, gates_new_imag,
            tokens_new_real, tokens_new_imag,
            SEQUENCE_LENGTH=T,
            CHANNELS=C,
            REVERSE=reverse,
        )

        return gates_new, tokens_new

    @staticmethod
    def setup_context(ctx, inputs: Tuple[torch.Tensor, torch.Tensor], outputs: Tuple[torch.Tensor, torch.Tensor]):
        # Define the context setup required for functorch transforms
        gates, tokens = inputs
        ctx.reverse = ScanBTC.reverse
        ctx.save_for_backward(gates, tokens, outputs[1])  # Example: saving inputs that may be needed later

    @staticmethod
    def backward(ctx, grad_gates: torch.Tensor, grad_tokens: torch.Tensor):
        # Given a linear scan: x_{t+1} = g_{t+1} * x_t + b_{t+1}
        # dL / dx_t = dL / dx_{t+1} * dx_{t+1}/dx_t = dL / dx_{t+1} * g*_{t+1}
        # grad_gates: dL / dg_t = dL / dx_{t+1} * x*_{t}
        # grad_tokens: dL / db_t = dL / dx_t
        # Padded reverse scan
        gates, tokens, output_tokens = ctx.saved_tensors
        B, T, C = tokens.shape

        assert grad_tokens.shape == tokens.shape

        reverse= ctx.reverse
        # Allocate output tensors
        d_tokens_real = torch.empty_like(tokens.real, dtype=tokens.real.dtype).contiguous()
        d_tokens_imag = torch.empty_like(tokens.imag, dtype=tokens.imag.dtype).contiguous()
        d_gates_real = torch.empty_like(gates.real, dtype=gates.real.dtype).contiguous()
        d_gates_imag = torch.empty_like(gates.imag, dtype=gates.imag.dtype).contiguous()

        # Split into real and imaginary parts
        gates_real = gates.real
        gates_imag = - gates.imag
        grad_tokens_real = grad_tokens.real
        grad_tokens_imag = grad_tokens.imag

        # Create shifted gates for backward pass
        if reverse:
            padded_shifted_gates_real = torch.cat([torch.ones_like(gates_real[..., :1, :]),gates_real,], dim=-2)[..., :-1, :].contiguous()
            padded_shifted_gates_imag = torch.cat([torch.zeros_like(gates_imag[..., :1, :]),gates_imag,], dim=-2)[..., :-1, :].contiguous()
        else:
            padded_shifted_gates_real = torch.cat([gates_real, torch.ones_like(gates_real[..., :1, :])], dim=-2)[..., 1:, :].contiguous()
            padded_shifted_gates_imag = torch.cat([gates_imag, torch.zeros_like(gates_imag[..., :1, :])], dim=-2)[..., 1:, :].contiguous()

        #print("bwd: padded_shifted_gates_real, padded_shifted_gates_imag, grad_tokens_real, grad_tokens_imag:")
        #print(padded_shifted_gates_real.mT, padded_shifted_gates_imag.mT, grad_tokens_real.mT, grad_tokens_imag.mT)
        # Backward scan
        backward_scan_btc_complex[(B, C)](
            padded_shifted_gates_real, padded_shifted_gates_imag,
            grad_tokens_real, grad_tokens_imag,
            d_gates_real, d_gates_imag,
            d_tokens_real, d_tokens_imag,
            SEQUENCE_LENGTH=T,
            CHANNELS=C,
            REVERSE=reverse,
        )
        #print('d_gates_real computed', d_gates_real.mT, d_gates_imag.mT)
        #print('d_tokens_real computed', d_tokens_real.mT, d_tokens_imag.mT)

        # grad_gates: dL / dg_t = dL / dx_{t+1} * x*_{t}   
        if ctx.reverse:
            padded_outputs = torch.cat(
                [output_tokens[..., 1:, :],
                torch.zeros_like(output_tokens[..., :1, :])], dim=-2)
        else:
            padded_outputs = torch.cat([torch.zeros_like(output_tokens[..., :1, :]), output_tokens], dim=-2)[..., :-1, :]
        d_gates =  torch.complex(d_tokens_real,d_tokens_imag) * padded_outputs.conj()

        # grad_tokens: dL / db_t = dL / dx_{t+1} * g*_{t+1}
        d_tokens = torch.complex(d_tokens_real, d_tokens_imag) # * gates.conj()

        return d_gates, d_tokens

    @staticmethod
    def vmap(info, in_dims: List, gates: torch.Tensor, tokens: torch.Tensor):
        """
        A "vmap rule" is the logic of how to perform the operation given inputs with one additional dimension.        
        
        Args:
             info: Metadata or additional info for vmap (not used here).
            in_dims: Tuple specifying the dimension to slice for each input.
            gates: Input gates tensor of shape (B, T, C) if in_dims[0] is None.
            tokens: Input tokens tensor of shape (B', B, T, C) if in_dims[1] is 0.
        Returns:
            outputs: Output tensor of shape (B', B, T, C).
        """    
        # Handle case where gates is constant (None in in_dims)
        if in_dims == (None, 0):
            # gates: (T, C) - constant for all batches  
            # tokens: (B, T, C) - vectorized over batch dimension
            # TODO: check:
            # tokens: (B, 1, L, C) → (B, L, C)
            tokens = tokens.squeeze(1)
            batch_size = tokens.shape[0]
            gates = gates.expand(batch_size,-1,-1).contiguous()
            outputs_g, outputs_a = ScanBTC.apply(gates, tokens)
            # Both outputs are batched
            return  (outputs_g, outputs_a), (None, 0)
        elif in_dims == (0, 0):
            assert gates.ndim == 3
            outputs_g , outputs_a= ScanBTC.apply(gates, tokens)
            return (outputs_g,outputs_a), (0,0)
        else:
            raise NotImplementedError(f"Only (None, 0, None) is implemented but requested {in_dims} gates{gates.shape} tokens {tokens.shape}")


class ScanBTC2(nn.Module):
    """
    Wrapper module around ScanBTC that allows setting the reverse flag cleanly.
    """
    def __init__(self, reverse: bool = False):
        super().__init__()
        self.reverse = reverse

    def forward(self, gates: torch.Tensor, tokens: torch.Tensor):
        # Dynamically create a subclass of ScanBTC with reverse captured
        ScanBTC.reverse = self.reverse
        return ScanBTC.apply(gates.contiguous(), tokens.contiguous())


def scan_btc_complex(x: torch.Tensor, y: torch.Tensor, reverse: bool=False):
    return ScanBTC2(reverse=reverse)(x, y)


@triton.jit
def forward_scan_btc_complex(
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
    REVERSE: tl.constexpr,
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
    if REVERSE:
        seq_idx = SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH)
    else:
        seq_idx = tl.arange(0, SEQUENCE_LENGTH)
    stride_idx = base_offset + seq_idx * CHANNELS

    # Remember we are dealing with complex numbers which are interleaved in memory
    stride_idx2 = stride_idx * 2

    # Return a tensor whose values are loaded from memory at located specified by the pointer
    tokens_r = tl.load(tokens_real + stride_idx2)
    tokens_i = tl.load(tokens_imag + stride_idx2)
    gates_r = tl.load(gates_real + stride_idx2)
    gates_i = tl.load(gates_imag + stride_idx2)

    #print("fwd in triton: gates_r, gates_i, tokens_r, tokens_i:")
    #print(gates_r, gates_i, tokens_r, tokens_i)
    # Perform scan operation
    tokens_new_r, tokens_new_i, gates_r_out, gates_i_out = tl.associative_scan(
        (tokens_r, tokens_i, gates_r, gates_i),
        axis=0,
        combine_fn=first_order_op_complex,
        reverse=False
    )

    out_idx = base_offset + seq_idx * CHANNELS
    out_idx2 = out_idx * 2

    # Store results
    tl.store(output_gates_real + out_idx2, gates_r_out)
    tl.store(output_gates_imag + out_idx2, gates_i_out)
    tl.store(output_tokens_real + out_idx2, tokens_new_r)
    tl.store(output_tokens_imag + out_idx2, tokens_new_i)

@triton.jit
def backward_scan_btc_complex(
    gates_real,
    gates_imag,
    grad_real,
    grad_imag,
    d_gates_real,
    d_gates_imag,
    d_tokens_real,
    d_tokens_imag,
    SEQUENCE_LENGTH: tl.constexpr,
    REVERSE: tl.constexpr,
    CHANNELS: tl.constexpr,
    ):
    """Backward scan with direct complex number handling."""

    # Program IDs for Batch and Channel (dimensions along which we parallelize)
    batch_id = tl.program_id(axis=0)  # Batch index
    channel_id = tl.program_id(axis=1)  # Channel index

    # Same as forward scan but read sequence in reverse order
    base_offset = batch_id * SEQUENCE_LENGTH * CHANNELS + channel_id
    if REVERSE:
        seq_idx =  tl.arange(0, SEQUENCE_LENGTH) 
        rev_idx =  (SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH)) 
    else:
        seq_idx = (SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH)) 
        rev_idx = tl.arange(0, SEQUENCE_LENGTH)
    strides = base_offset + seq_idx * CHANNELS
    reversed_strides = base_offset + rev_idx* CHANNELS

    strides2 = strides*2
    reversed_strides2 = reversed_strides*2

    # Remember we are dealing with complex numbers which are interleaved in memory

    # Load data in reverse order
    gates_r = tl.load(gates_real + strides)
    gates_i = tl.load(gates_imag + strides)
    grad_r = tl.load(grad_real + strides2)
    grad_i = tl.load(grad_imag + strides2)

    #print("bwd in triton:gates_r, gates_i, grad_r, grad_i:")
    #print(gates_r, gates_i, grad_r, grad_i)
    
    # Perform backward scan
    tokens_new_r, tokens_new_i, gates_r_out, gates_i_out = tl.associative_scan(
        (grad_r, grad_i, gates_r, gates_i),
        axis=0,
        combine_fn=first_order_op_complex,
        reverse=False,
    )
    # Store results in reverse order
    #print("bwd out triton:gates_r_out, gates_i_out, tokens_new_r, tokens_new_i:")
    #print(gates_r_out, gates_i_out, tokens_new_r,tokens_new_i)
    if REVERSE:
        seq_idx = (SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH)) 
        rev_idx = tl.arange(0, SEQUENCE_LENGTH) 
    else:
        seq_idx =  tl.arange(0, SEQUENCE_LENGTH)
        rev_idx = (SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH)) 

    store_idx = base_offset + rev_idx * CHANNELS
    
    # when we store we need to reverse the strides
    tl.store(d_tokens_real + store_idx, tokens_new_r)
    tl.store(d_tokens_imag + store_idx, tokens_new_i)
    tl.store(d_gates_real + store_idx, gates_r_out)
    tl.store(d_gates_imag + store_idx, gates_i_out)


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
      A_out = A_j * A_i
      Bu_out = A_j * b_i + b_j
  """
  A_i, b_i = q_i
  A_j, b_j = q_j

  return A_j * A_i,  torch.addcmul(b_j, A_j, b_i)

class ComplexLinearScanBTC(torch.autograd.Function):
    @staticmethod
    def forward(
        gates: torch.Tensor,
        tokens: torch.Tensor,
        reverse: bool = False):
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

        if reverse:
            gates = torch.flip(gates, dims=[-2])
            tokens = torch.flip(tokens, dims=[-2])

        scanned_gates = torch.zeros_like(gates, dtype=torch.cfloat)
        scanned_tokens = torch.zeros_like(tokens, dtype=torch.cfloat)
        
        # initialize x_0 = a_0 * x_0 + b_prefix,  but we assume x_0 = 0
        scanned_gates[..., 0, :] = gates[..., 0, :]
        scanned_tokens[..., 0, :] = tokens[..., 0, :]

        # Initialize (a_prefix, b_prefix) variables
        current_gates = gates[..., 0, :]
        current_tokens = tokens[..., 0, :]

    
        for t in range(1, seq_len):
            #outputs[..., t] = gates[..., t] * outputs[..., t-1] + tokens[..., t]

            # (a_prefix, b_prefix) = (a_prefix, b_prefix) o (gates[..., t], tokens[..., t])
            current_gates, current_tokens = binary_operator((current_gates, current_tokens), (gates[..., t, :], tokens[..., t, :]))
            # x_t = a_prefix * x_0 + b_prefix
            scanned_gates[..., t, :] = current_gates
            scanned_tokens[..., t, :] = current_tokens
        
        if reverse:
            scanned_gates = torch.flip(scanned_gates, dims=[-2])
            scanned_tokens = torch.flip(scanned_tokens, dims=[-2])

        return scanned_gates, scanned_tokens
        
    @staticmethod
    def backward(ctx, grads_gates: torch.Tensor, grad_outputs: torch.Tensor):
        """
        Backward pass for the complex linear scan.
        Args:
            grad_outputs: Complex tensor of shape (batch, dim, len).
        Returns:
            Gradients for tokens, gates, and contributions (all complex tensors).
        """
        gates, tokens, outputs = ctx.saved_tensors
        seq_len = tokens.shape[-2]

        reverse = ctx.reverse

        if reverse:
            gates = torch.flip(gates, dims=[-2])
            tokens = torch.flip(tokens, dims=[-2])
            grad_outputs = torch.flip(grad_outputs, dims=[-2])
            outputs = torch.flip(outputs, dims=[-2])

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

        if reverse:
            grad_tokens = grad_tokens.flip(dims=[-2])
            grad_gates = grad_gates.flip(dims=[-2])

        return grad_gates, grad_tokens, None

    @staticmethod
    def setup_context(ctx, inputs: Tuple[torch.Tensor,torch.Tensor,torch.Tensor], outputs: Tuple[torch.Tensor,torch.Tensor]):
        # Define the context setup required for functorch transforms
        gates, tokens, reverse = inputs
        _, output_tokens = outputs
        ctx.save_for_backward(gates, tokens, output_tokens)  # Example: saving inputs that may be needed later
        ctx.reverse = reverse

    @staticmethod
    def vmap(info, in_dims: List, gates: torch.Tensor, tokens: torch.Tensor, reverse: bool =False):
        """
        A "vmap rule" is the logic of how to perform the operation given inputs with one additional dimension.        
        with vmap we slice over dimension and applies function to each slide:

        Args:
             info: Metadata or additional info for vmap (not used here).
            in_dims: Tuple specifying the dimension to slice for each input.
            gates: Input gates tensor of shape (B, T, C) if in_dims[0] is None.
            tokens: Input tokens tensor of shape (B', B, T, C) if in_dims[1] is 0.
        Returns:
            outputs: Output tensor of shape (B', B, T, C).
            out_dims: Tuple specifying the dimension to slice for each output.
        """

        # Handle case where gates is constant (None in in_dims)
        if in_dims == (None, 0, None):
            # None for the first input: this argument is not batched but the same across all dimensions
            # 0 for the second input: 
            # gates: (T, C) - constant for all batches
            # tokens: (B, T, C) - vectorized over batch dimension
            batch_size = tokens.shape[0]
            gates_expanded = gates.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            outputs_g, outputs_k = ComplexLinearScanBTC.apply(gates_expanded, tokens, reverse)
            # Return with correct out_dims - both outputs are batched
            out_dims = (0, 0)
            return (outputs_g, outputs_k), out_dims
        elif in_dims == (0, 0, None):
            # gates: (B, T, C) -
            # tokens: (B, T, C) - 
            # output to be T, C
            outputs_g, outputs_k = ComplexLinearScanBTC.apply(gates, tokens, reverse)
            out_dims = (None, None)
            return (outputs_g, outputs_k), out_dims
        else:
            raise NotImplementedError(f"Only (None, 0) is implemented -- in_dims was {in_dims}")


#@torch.compiler.disable(recursive=False)
def scan_btc_complex_autograd(gates: torch.Tensor, tokens: torch.Tensor, reverse: bool =False):
    gates = gates.contiguous()
    tokens = tokens.contiguous()
    return ComplexLinearScanBTC.apply(gates, tokens, reverse)


class ComplexLinearScanBCT(torch.autograd.Function):
    @staticmethod
    def forward(gates, tokens, reverse=False):
        """
        Forward pass for the complex linear scan.
        x_t = g_t * x_{t-1} + b_t
        Args:
            gates: Complex tensor of shape (batch, dim, len).
            tokens: Complex tensor of shape (batch, dim, len).
        Returns:
            outputs: Complex tensor of shape (batch, dim, len).
        """
        seq_len = tokens.shape[-1]
        outputs = torch.zeros_like(tokens, dtype=torch.cfloat)
        outputs_gates = torch.zeros_like(gates, dtype=torch.cfloat)

        if reverse:
            gates = torch.flip(gates, dims=[-1])
            tokens = torch.flip(tokens, dims=[-1])

        # Initialize (a_prefix, b_prefix) variables
        a_prefix = gates[..., 0]
        b_prefix = tokens[..., 0]

        # initialize x_0 = a_0 * x_0 + b_prefix,  but we assume x_0 = 0
        outputs[..., 0] = tokens[..., 0]
        outputs_gates[..., 0] = gates[..., 0]

        for t in range(1, seq_len):
            #outputs[..., t] = gates[..., t] * outputs[..., t-1] + tokens[..., t]

            # (a_prefix, b_prefix) = (a_prefix, b_prefix) o (gates[..., t], tokens[..., t])
            a_prefix, b_prefix = binary_operator((a_prefix, b_prefix), (gates[..., t], tokens[..., t]))
            # x_t = a_prefix * x_0 + b_prefix
            outputs[..., t] = b_prefix
            outputs_gates[..., t] = a_prefix

        if reverse:
            outputs_gates = torch.flip(outputs_gates, dims=[-1])
            outputs = torch.flip(outputs, dims=[-1])


        return outputs_gates, outputs

        
    @staticmethod
    def backward(ctx, grad_gates, grad_outputs):
        """
        Backward pass for the complex linear scan.
        Args:
            grad_outputs: Complex tensor of shape (batch, dim, len).
        Returns:
            Gradients for tokens, gates, and contributions (all complex tensors).
        """
        gates,tokens, outputs = ctx.saved_tensors
        seq_len = tokens.shape[-1]

        reverse = ctx.reverse

        if reverse:
            gates = torch.flip(gates, dims=[-1])
            tokens = torch.flip(tokens, dims=[-1])
            grad_outputs = torch.flip(grad_outputs, dims=[-1])
            #grad_gates = torch.flip(grad_gates, dims=[-1])
            outputs = torch.flip(outputs, dims=[-1])

        # Initialize gradients
        grad_gates = torch.zeros_like(gates, dtype=torch.cfloat)
        grad_tokens = torch.zeros_like(tokens, dtype=torch.cfloat)

        # Gradient of the loss with respect to the last output
        padded_shifted_gates = torch.cat([gates, torch.ones_like(gates[..., :1])], dim=-1)[..., 1:].contiguous().conj()

        padded_shifted_gates_rev = torch.flip(padded_shifted_gates, dims=[-1])
        grad_outputs_rev = torch.flip(grad_outputs, dims=[-1])
        #print("padded_shifted_gates_rev", padded_shifted_gates_rev)
        #print("grad_outputs_rev", grad_outputs_rev)

        # Forward scan
        grad_tokens[..., 0] = grad_outputs_rev[..., 0]
        a_prefix = padded_shifted_gates_rev[..., 0]
        b_prefix = grad_outputs_rev[..., 0]

        for t in range(1, seq_len):
            # (a_prefix, b_prefix) = (a_prefix, b_prefix) o (shifted_gates[..., t], grads[..., t])
            a_prefix, b_prefix = binary_operator((a_prefix, b_prefix), (padded_shifted_gates_rev[..., t], grad_outputs_rev[..., t]))
            grad_tokens[..., t] = b_prefix

        padded_outputs = torch.cat([torch.zeros_like(outputs[..., :1]), outputs], dim=-1)[..., :-1]

        grad_tokens = torch.flip(grad_tokens, dims=[-1])
        grad_gates = grad_tokens * padded_outputs.conj()

        #print('padded_outputs', padded_outputs)
        if reverse:
            grad_tokens = grad_tokens.flip(dims=[-1])
            grad_gates = grad_gates.flip(dims=[-1])

        #print('out_grad_tokens', grad_tokens)
        return grad_gates, grad_tokens, None

    @staticmethod
    def vmap(info, in_dims, gates, tokens, reverse=False):
        """
        A "vmap rule" is the logic of how to perform the operation given inputs with one additional dimension.        
        
        Args:
             info: Metadata or additional info for vmap (not used here).
            in_dims: Tuple specifying the dimension to slice for each input.
            gates: Input gates tensor of shape (B, T, C) if in_dims[0] is None.
            tokens: Input tokens tensor of shape (B', B, T, C) if in_dims[1] is 0.
        Returns:
            outputs: Output tensor of shape (B', B, T, C).
        """

        # Handle case where gates is constant (None in in_dims)
        if in_dims == (None, 0, None):
            # gates: (T, C) - constant for all batches
            # tokens: (B, T, C) - vectorized over batch dimension
            batch_size = tokens.shape[0]
            gates_expanded = gates.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            outputs_g, outputs_k = ComplexLinearScanBCT.apply(gates_expanded, tokens, reverse)
            out_dims = (0, 0)
        else:
            raise NotImplementedError("Only (None, 0) is implemented")

        return (outputs_g, outputs_k), out_dims
    
    def setup_context(ctx, inputs, outputs):
        gates, tokens, reverse = inputs
        _, output_tokens = outputs
        ctx.save_for_backward(gates, tokens, output_tokens)
        ctx.reverse = reverse

def scan_bct_complex_autograd(gates, tokens, reverse=False):
    """
    Solve a first-order recurrence relation for complex numbers.
    x_t = a_t x_{t-1} + b_t
    where :math:`a_t` ("gates") and :math:`b_t` ("tokens") are sequences of vectors.

    """
    gates = gates.contiguous()
    tokens = tokens.contiguous()
    return ComplexLinearScanBCT.apply(gates, tokens, reverse)

@triton.jit
def pack128(a_real, a_imag, b_real, b_imag):
    """Fixed version of pack128 - pack 4 float32 values into 2 uint64 values"""
    a_real_u32 = a_real.to(tl.uint32, bitcast=True)
    a_imag_u32 = a_imag.to(tl.uint32, bitcast=True)
    b_real_u32 = b_real.to(tl.uint32, bitcast=True)
    b_imag_u32 = b_imag.to(tl.uint32, bitcast=True)

    # Pack first complex number (a_real, a_imag) into low64
    low64 = a_real_u32.to(tl.uint64) | (a_imag_u32.to(tl.uint64) << 32)
    # Pack second complex number (b_real, b_imag) into high64  
    high64 = b_real_u32.to(tl.uint64) | (b_imag_u32.to(tl.uint64) << 32)
    
    return low64, high64

@triton.jit
def unpack128(low64, high64):
    """Fixed version of unpack128 - unpack 2 uint64 values into 4 float32 values"""
    # Unpack first complex number from low64
    a_real = (low64 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    a_imag = (low64 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    
    # Unpack second complex number from high64
    b_real = (high64 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b_imag = (high64 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    
    return a_real, a_imag, b_real, b_imag

@triton.jit
def first_order_op_128(l0, l1, r0, r1):
    a_r, a_i, b_r, b_i = unpack128(l0, l1)
    g_r, g_i, t_r, t_i = unpack128(r0, r1)

    new_a_r, new_a_i = complex_mul(g_r, g_i, a_r, a_i)
    mul_r, mul_i = complex_mul(g_r, g_i, b_r, b_i)

    new_b_r = mul_r + t_r
    new_b_i = mul_i + t_i

    return pack128(new_a_r, new_a_i, new_b_r, new_b_i)

@triton.jit
def forward_scan_bct_complex_packed(
    gates_real,
    gates_imag,
    tokens_real,
    tokens_imag,
    output_gates_real,
    output_gates_imag,
    output_tokens_real,
    output_tokens_imag,
    SEQUENCE_LENGTH: tl.constexpr,
    REVERSE: tl.constexpr,
    CHANNELS: tl.constexpr,
):
    # Program IDs for Batch and Channel (dimensions along which we parallelize)
    batch_id = tl.program_id(axis=0)  # Batch index
    channel_id = tl.program_id(axis=1)  # Channel index

    # Base offset for batch and channel in the BCT layout
    base_offset = batch_id * CHANNELS * SEQUENCE_LENGTH + channel_id * SEQUENCE_LENGTH

    # Sequence strides: Stride along the sequence dimension (T)
    if REVERSE:
        seq_idx = SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH)
    else:
        seq_idx = tl.arange(0, SEQUENCE_LENGTH)

    strides = base_offset + seq_idx

    # we are dealing with complex numbers with real and imag parts interleaved in memory
    strides = strides * 2

    # Return a tensor whose values are loaded from memory at located specified by the pointer
    tokens_r = tl.load(tokens_real + strides)
    tokens_i = tl.load(tokens_imag + strides)
    gates_r = tl.load(gates_real + strides)
    gates_i = tl.load(gates_imag + strides)

    # Pack and scan
    l0, l1 = pack128(gates_r, gates_i, tokens_r, tokens_i)
    out0, out1 = tl.associative_scan((l0, l1), axis=0, combine_fn=first_order_op_128)
    gates_r_out, gates_i_out, tokens_new_r, tokens_new_i = unpack128(out0, out1)

    # Store results
    tl.store(output_gates_real + strides, gates_r_out)
    tl.store(output_gates_imag + strides, gates_i_out)
    tl.store(output_tokens_real + strides, tokens_new_r)
    tl.store(output_tokens_imag + strides, tokens_new_i)

@triton.jit
def forward_scan_bct_complex_split(
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
    REVERSE: tl.constexpr,
    ):
    """
    Forward scan for split real/imaginary tensors (not interleaved).
    """
    # Program IDs for Batch and Channel (dimensions along which we parallelize)
    batch_id = tl.program_id(axis=0)  # Batch index
    channel_id = tl.program_id(axis=1)  # Channel index

    # Base offset for batch and channel in the BCT layout
    base_offset = batch_id * CHANNELS * SEQUENCE_LENGTH + channel_id * SEQUENCE_LENGTH

    # Sequence strides: Stride along the sequence dimension (T)
    if REVERSE:
        seq_idx = SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH)
    else:
        seq_idx = tl.arange(0, SEQUENCE_LENGTH)

    strides = base_offset + seq_idx
    # Note: No *2 here since real/imag are separate tensors, not interleaved

    # Load from separate real and imaginary tensors
    tokens_r = tl.load(tokens_real + strides)
    tokens_i = tl.load(tokens_imag + strides)
    gates_r = tl.load(gates_real + strides)
    gates_i = tl.load(gates_imag + strides)

    # Scan outputs
    tokens_new_r, tokens_new_i, gates_r_out, gates_i_out = tl.associative_scan(
        (tokens_r, tokens_i, gates_r, gates_i),
        axis=0,
        combine_fn=first_order_op_complex,
        reverse=False
    )

    # Store results
    tl.store(output_gates_real + strides, gates_r_out)
    tl.store(output_gates_imag + strides, gates_i_out)
    tl.store(output_tokens_real + strides, tokens_new_r)
    tl.store(output_tokens_imag + strides, tokens_new_i)

@triton.jit
def backward_scan_bct_complex_split(
    gates_real,
    gates_imag,
    grad_real,
    grad_imag,
    d_gates_real,
    d_gates_imag,
    d_tokens_real,
    d_tokens_imag,
    SEQUENCE_LENGTH: tl.constexpr,
    REVERSE: tl.constexpr,
    CHANNELS: tl.constexpr,
    ):
    """Backward scan for split real/imaginary tensors (not interleaved)."""
    # Program IDs for Batch and Channel (dimensions along which we parallelize)
    batch_id = tl.program_id(axis=0)  # Batch index
    channel_id = tl.program_id(axis=1)  # Channel index

    base_offset = batch_id * CHANNELS * SEQUENCE_LENGTH + channel_id * SEQUENCE_LENGTH

    # For backward scan, read in reverse order
    if REVERSE:
        seq_idx = tl.arange(0, SEQUENCE_LENGTH)
    else:
        seq_idx = (SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH))
    
    reversed_strides = base_offset + seq_idx
    
    # Load data in reverse order for backward scan
    # Note: No *2 here since real/imag are separate tensors
    gates_r = tl.load(gates_real + reversed_strides)
    gates_i = tl.load(gates_imag + reversed_strides)
    grad_r = tl.load(grad_real + reversed_strides)
    grad_i = tl.load(grad_imag + reversed_strides)
    
    # Perform backward scan
    tokens_new_r, tokens_new_i, gates_r_out, gates_i_out = tl.associative_scan(
        (grad_r, grad_i, gates_r, gates_i),
        axis=0,
        combine_fn=first_order_op_complex,
        reverse=False,
    )

    if REVERSE:
        rev_idx = tl.arange(0, SEQUENCE_LENGTH) 
    else:
        rev_idx = (SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH)) 
    store_idx = base_offset + rev_idx 
    tl.store(d_tokens_real + store_idx, tokens_new_r)
    tl.store(d_tokens_imag + store_idx, tokens_new_i)
    tl.store(d_gates_real + store_idx, gates_r_out)
    tl.store(d_gates_imag + store_idx, gates_i_out)

@triton.jit
def backward_scan_bct_complex_packed(
    gates_real,
    gates_imag,
    grad_real,
    grad_imag,
    d_tokens_real,
    d_tokens_imag,
    d_gates_real,
    d_gates_imag,
    SEQUENCE_LENGTH: tl.constexpr,
    REVERSE: tl.constexpr,
    CHANNELS: tl.constexpr,
    GATES_STRIDE: tl.constexpr,      # 1 for separate tensors, 2 for complex views
    GRAD_STRIDE: tl.constexpr,       # 1 for separate tensors, 2 for complex views
    OUTPUT_STRIDE: tl.constexpr,     # 1 for separate tensors, 2 for complex views
):
    batch_id = tl.program_id(axis=0)
    channel_id = tl.program_id(axis=1)

    base_offset = batch_id * CHANNELS * SEQUENCE_LENGTH + channel_id * SEQUENCE_LENGTH

    # For backward scan, read in reverse order
    if REVERSE:
        seq_idx = tl.arange(0, SEQUENCE_LENGTH)
    else:
        seq_idx = (SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH))
    
    # Apply appropriate strides for different memory layouts
    gates_strides = (base_offset + seq_idx) * GATES_STRIDE
    grad_strides = (base_offset + seq_idx) * GRAD_STRIDE
    
    # Load data in reverse order for backward scan
    # Note: gates and grads have different memory layouts
    gates_r = tl.load(gates_real + gates_strides)
    gates_i = tl.load(gates_imag + gates_strides)
    grad_r = tl.load(grad_real + grad_strides)
    grad_i = tl.load(grad_imag + grad_strides)
    
    # Perform backward scan - note the order matches first_order_op_complex
    l0, l1 = pack128(gates_r, gates_i, grad_r, grad_i)
    out0, out1 = tl.associative_scan((l0, l1), axis=0, combine_fn=first_order_op_128)
    gates_r_out, gates_i_out, tokens_new_r, tokens_new_i = unpack128(out0, out1)

    if REVERSE:
        rev_idx = tl.arange(0, SEQUENCE_LENGTH) 
    else:
        rev_idx = (SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH)) 

    output_strides = (base_offset + rev_idx) * OUTPUT_STRIDE
    tl.store(d_tokens_real + output_strides, tokens_new_r)
    tl.store(d_tokens_imag + output_strides, tokens_new_i)
    tl.store(d_gates_real + output_strides, gates_r_out)
    tl.store(d_gates_imag + output_strides, gates_i_out)


class ScanBCTPacked(torch.autograd.Function):
    @staticmethod
    def forward(gates, tokens, reverse=False):
        B, C, T = gates.shape
        assert tokens.shape == (B, C, T), "Tokens must have the same shape as gates."
        assert gates.is_contiguous(), "Gates tensor must be contiguous."
        assert tokens.is_contiguous(), "Tokens tensor must be contiguous."

        # Split into real and imaginary parts
        gates_real = gates.real
        gates_imag = gates.imag
        tokens_real = tokens.real
        tokens_imag = tokens.imag

        # Allocate output tensors
        gates_new = torch.empty_like(gates, dtype=gates.dtype)
        tokens_new = torch.empty_like(tokens, dtype=tokens.dtype)

        gates_new_real = gates_new.real
        gates_new_imag = gates_new.imag
        tokens_new_real = tokens_new.real
        tokens_new_imag = tokens_new.imag

        # Forward pass
        forward_scan_bct_complex_packed[(B, C)](
            gates_real, gates_imag,
            tokens_real, tokens_imag,
            gates_new_real, gates_new_imag,
            tokens_new_real, tokens_new_imag,
            SEQUENCE_LENGTH=T,
            REVERSE=reverse,
            CHANNELS=C,
        )
        return gates_new, tokens_new

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # Define the context setup required for functorch transforms
        gates, tokens, reverse = inputs
        _, output_tokens = outputs
        ctx.save_for_backward(gates, tokens, output_tokens)  # Example: saving inputs that may be needed later
        ctx.reverse = reverse

    @staticmethod
    def backward(ctx, grad_gates, grad_tokens):
        gates, tokens, output_tokens = ctx.saved_tensors
        reverse = ctx.reverse
        B, C, T = gates.shape
        
        # Allocate output tensors
        # OPTION 1: Use complex tensor views (now supported!)
        d_gates = torch.empty_like(gates)
        d_tokens = torch.empty_like(tokens)
        
        d_gates_real = d_gates.real
        d_gates_imag = d_gates.imag
        d_tokens_real = d_tokens.real
        d_tokens_imag = d_tokens.imag

        # Split inputs
        grad_tokens_real = grad_tokens.real
        grad_tokens_imag = grad_tokens.imag

        # Create shifted gates
        if reverse:
            d_gates_real[..., 1:] = gates.real[..., :-1]
            d_gates_real[..., 0] = 1.0
            d_gates_imag[..., 1:] = -gates.imag[..., :-1]
            d_gates_imag[..., 0] = 0.0
        else:
            d_gates_real[..., :-1] = gates.real[..., 1:]
            d_gates_real[..., -1] = 1.0
            d_gates_imag[..., :-1] = -gates.imag[..., 1:]
            d_gates_imag[..., -1] = 0.0

        # Determine stride based on tensor layout
        #gates_stride = 2 if gates_real.stride()[-1] == 2 else 1
        grad_stride = 2 if grad_tokens_real.stride()[-1] == 2 else 1  
        output_stride = 2 if d_tokens_real.stride()[-1] == 2 else 1

        # Call strided kernel
        backward_scan_bct_complex_packed[(B, C)](
            d_gates_real, d_gates_imag,
            grad_tokens_real, grad_tokens_imag,
            d_tokens_real, d_tokens_imag,
            d_gates_real, d_gates_imag,
            SEQUENCE_LENGTH=T,
            REVERSE=reverse,
            CHANNELS=C,
            GATES_STRIDE=2,
            GRAD_STRIDE=grad_stride,
            OUTPUT_STRIDE=output_stride,
        )
        
        # Compute final gate gradients
        if reverse:
            d_gates_real[..., :-1] = (d_tokens_real[..., :-1] * output_tokens.real[..., 1:] + 
                                      d_tokens_imag[..., :-1] * output_tokens.imag[..., 1:])
            d_gates_imag[..., :-1] = (d_tokens_imag[..., :-1] * output_tokens.real[..., 1:] - 
                                      d_tokens_real[..., :-1] * output_tokens.imag[..., 1:])
            d_gates_real[..., -1] = 0.0
            d_gates_imag[..., -1] = 0.0
        else:
            d_gates_real[..., 1:] = (d_tokens_real[..., 1:] * output_tokens.real[..., :-1] + 
                                     d_tokens_imag[..., 1:] * output_tokens.imag[..., :-1])
            d_gates_imag[..., 1:] = (d_tokens_imag[..., 1:] * output_tokens.real[..., :-1] - 
                                     d_tokens_real[..., 1:] * output_tokens.imag[..., :-1])
            d_gates_real[..., 0] = 0.0
            d_gates_imag[..., 0] = 0.0

        return d_gates, d_tokens, None

    
    @staticmethod
    def vmap(info, in_dims, gates, tokens, reverse=False):
        """
        A "vmap rule" is the logic of how to perform the operation given inputs with one additional dimension.        
        
        Args:
             info: Metadata or additional info for vmap (not used here).
            in_dims: Tuple specifying the dimension to slice for each input.
            gates: Input gates tensor of shape (B, T, C) if in_dims[0] is None.
            tokens: Input tokens tensor of shape (B', B, T, C) if in_dims[1] is 0.
        Returns:
            outputs: Output tensor of shape (B', B, T, C).
        """

        # Handle case where gates is constant (None in in_dims)
        if in_dims == (None, 0, None):
            # gates: (C, T) - constant across batch
            # tokens: (B, C, T) - vectorized over batch dimension
            B = tokens.shape[0]
            # Expand gates to match batch dimension
            gates_expanded = gates.unsqueeze(0).expand(B, -1, -1).contiguous()
            outputs_g, outputs_k = ScanBCT.apply(gates_expanded, tokens, reverse)
            out_dims = (0, 0)  # Both outputs vectorized over batch dimension
        else:
            raise NotImplementedError("Only (None, 0) is implemented")

        return (outputs_g, outputs_k), out_dims


# this function operates directly on complex which has compilation issues
#@torch.compiler.disable(recursive=True)
def scan_bct_complex_packed(gates, tokens, reverse=False):
    return ScanBCTPacked.apply(gates, tokens, reverse)


class ScanBCTPackedSplit(torch.autograd.Function):
    @staticmethod
    def forward(gates_real, gates_imag, tokens_real, tokens_imag, reverse=False):
        B, C, T = gates_real.shape
        assert tokens_real.shape == (B, C, T), "Tokens must have the same shape as gates."
        assert gates_real.is_contiguous(), "Gates real tensor must be contiguous."
        assert gates_imag.is_contiguous(), "Gates imag tensor must be contiguous."
        assert tokens_real.is_contiguous(), "Tokens real tensor must be contiguous."
        assert tokens_imag.is_contiguous(), "Tokens imag tensor must be contiguous."

        # Allocate output tensors
        gates_new_real = torch.empty_like(gates_real)
        gates_new_imag = torch.empty_like(gates_imag)
        tokens_new_real = torch.empty_like(tokens_real)
        tokens_new_imag = torch.empty_like(tokens_imag)
 
        forward_scan_bct_complex_split[(B, C)](
            gates_real, gates_imag,           # Direct use of split inputs
            tokens_real, tokens_imag,         # Direct use of split inputs
            gates_new_real, gates_new_imag,   # Direct output to split tensors
            tokens_new_real, tokens_new_imag, # Direct output to split tensors
            SEQUENCE_LENGTH=T,
            CHANNELS=C,
            REVERSE=reverse,
        )
        
        return gates_new_real, gates_new_imag, tokens_new_real, tokens_new_imag

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        gates_real, gates_imag, tokens_real, tokens_imag, reverse = inputs
        _, _, output_tokens_real, output_tokens_imag = outputs
        ctx.save_for_backward(gates_real, gates_imag, tokens_real, tokens_imag, output_tokens_real, output_tokens_imag)
        ctx.reverse = reverse

    @staticmethod
    def backward(ctx, grad_gates_real, grad_gates_imag, grad_tokens_real, grad_tokens_imag):
        gates_real, gates_imag, tokens_real, tokens_imag, output_tokens_real, output_tokens_imag = ctx.saved_tensors
        reverse = ctx.reverse
        B, C, T = gates_real.shape

        # Allocate output tensors
        d_tokens_real = torch.empty_like(tokens_real)
        d_tokens_imag = torch.empty_like(tokens_imag)
        d_gates_real = torch.empty_like(gates_real)
        d_gates_imag = torch.empty_like(gates_imag)

        # Apply conjugation to split tensors directly (conj flips sign of imaginary part)
        gates_imag_conj = gates_imag

        # Create shifted gates for backward pass 
        if reverse:
            padded_shifted_gates_real = torch.cat([torch.ones_like(gates_real[..., :1]), gates_real], dim=-1)[..., :-1].contiguous()
            padded_shifted_gates_imag = torch.cat([torch.zeros_like(gates_imag_conj[..., :1]), gates_imag_conj], dim=-1)[..., :-1].contiguous()
        else:
            padded_shifted_gates_real = torch.cat([gates_real, torch.ones_like(gates_real[..., :1])], dim=-1)[..., 1:].contiguous()
            padded_shifted_gates_imag = torch.cat([gates_imag_conj, torch.zeros_like(gates_imag_conj[..., :1])], dim=-1)[..., 1:].contiguous()

        backward_scan_bct_complex[(B, C)](
            padded_shifted_gates_real, padded_shifted_gates_imag,
            grad_tokens_real, grad_tokens_imag,
            d_gates_real, d_gates_imag,
            d_tokens_real, d_tokens_imag,
            SEQUENCE_LENGTH=T,
            REVERSE=reverse,
            CHANNELS=C,
        )
           
        # Compute d_gates using split tensor operations
        if reverse:
            padded_outputs_real = torch.cat(
                [output_tokens_real[..., 1:], torch.zeros_like(output_tokens_real[..., :1])], dim=-1
            )
            padded_outputs_imag = torch.cat(
                [output_tokens_imag[..., 1:], torch.zeros_like(output_tokens_imag[..., :1])], dim=-1
            )
        else:
            padded_outputs_real = torch.cat(
                [torch.zeros_like(output_tokens_real[..., :1]), output_tokens_real], dim=-1
            )[..., :-1]
            padded_outputs_imag = torch.cat(
                [torch.zeros_like(output_tokens_imag[..., :1]), output_tokens_imag], dim=-1
            )[..., :-1]

        # Fused complex multiplication with conjugation
        # d_tokens * padded_outputs.conj() = d_tokens * (real - imag*i)
        d_gates_real_final = torch.addcmul(
            d_tokens_real * padded_outputs_real,  # Real part
            d_tokens_imag, padded_outputs_imag    # + imag * imag
        )
        d_gates_imag_final = torch.addcmul(
            d_tokens_imag * padded_outputs_real,  # Imag part  
            d_tokens_real, padded_outputs_imag, value=-1  # - real * imag
        )

        return d_gates_real_final, d_gates_imag_final, d_tokens_real, d_tokens_imag, None

    @staticmethod
    def vmap(info, in_dims, gates_real, gates_imag, tokens_real, tokens_imag, reverse=False):
        """
        A "vmap rule" is the logic of how to perform the operation given inputs with one additional dimension.        
        
        Args:
             info: Metadata or additional info for vmap (not used here).
            in_dims: Tuple specifying the dimension to slice for each input.
            gates_real: Input gates real tensor 
            gates_imag: Input gates imag tensor
            tokens_real: Input tokens real tensor
            tokens_imag: Input tokens imag tensor
        Returns:
            outputs: Output tensors.
        """

        # Handle case where gates is constant (None in in_dims)
        if in_dims == (None, None, 0, 0, None):
            # gates: (C, T) - constant across batch
            # tokens: (B, C, T) - vectorized over batch dimension
            B = tokens_real.shape[0]
            # Expand gates to match batch dimension
            gates_real_expanded = gates_real.unsqueeze(0).expand(B, -1, -1).contiguous()
            gates_imag_expanded = gates_imag.unsqueeze(0).expand(B, -1, -1).contiguous()
            outputs_gr, outputs_gi, outputs_tr, outputs_ti = ScanBCTPackedSplit.apply(
                gates_real_expanded, gates_imag_expanded, tokens_real, tokens_imag, reverse)
            out_dims = (0, 0, 0, 0)  # All outputs vectorized over batch dimension
        else:
            raise NotImplementedError(f"Only (None, None, 0, 0, None) is implemented but requested {in_dims}")

        return (outputs_gr, outputs_gi, outputs_tr, outputs_ti), out_dims


@torch.jit.script
def scan_bct_complex_split_compiled(
    gr: torch.Tensor,
    gi: torch.Tensor,
    tr: torch.Tensor,
    ti: torch.Tensor,
    reverse: bool = False):
    out_gr, out_gi, out_tr, out_ti = ScanBCTPackedSplit.apply(gr, gi, tr, ti, reverse)
    return out_gr, out_gi, out_tr, out_ti


@torch.compiler.disable(recursive=False)
def scan_bct_complex_split(
    gates: torch.Tensor,
    tokens: torch.Tensor,
    reverse: bool =False) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: fix packing so that we can feed in values as float32s
    # Torchinductor does not support code generation for complex operators.
    gr, gi = gates.real.contiguous(), gates.imag.contiguous()
    tr, ti = tokens.real.contiguous(), tokens.imag.contiguous()
    out_gr, out_gi, out_tr, out_ti = scan_bct_complex_split_compiled(gr, gi, tr, ti, reverse)
    return _recombine_complex(out_gr, out_gi, out_tr, out_ti)

@torch.jit.script
def _recombine_complex(
    gr: torch.Tensor,
    gi: torch.Tensor,
    tr: torch.Tensor,
    ti: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.complex(gr, gi), torch.complex(tr, ti)

scan_bct_complex_split_fused = scan_bct_complex_split