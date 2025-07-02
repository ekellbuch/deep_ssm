"""

"""
import torch
import os
from typing import Tuple
try:
    import triton
    import triton.language as tl
    load_triton = True
except:
    load_triton = False


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
        gates_real, gates_imag = gates.real, gates.imag
        tokens_real, tokens_imag = tokens.real, tokens.imag

        # Allocate output tensors
        gates_new = torch.zeros_like(gates, dtype=gates.dtype)
        tokens_new = torch.zeros_like(tokens, dtype=tokens.dtype)

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
            enable_fp_fusion=False
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
        d_tokens_real = torch.zeros_like(tokens.real, dtype=tokens.real.dtype)
        d_tokens_imag = torch.zeros_like(tokens.imag, dtype=tokens.imag.dtype)
        d_gates_real = torch.zeros_like(gates.real, dtype=gates.real.dtype)
        d_gates_imag = torch.zeros_like(gates.imag, dtype=gates.imag.dtype)

        # Split into real and imaginary parts
        gates = gates.conj()
        gates_real, gates_imag = gates.real, gates.imag
        grad_tokens_real, grad_tokens_imag = grad_tokens.real, grad_tokens.imag

        # Create shifted gates for backward pass 
        if reverse:
            padded_shifted_gates_real = torch.cat([torch.ones_like(gates_real[..., :1]), gates_real], dim=-1)[..., :-1].contiguous()
            padded_shifted_gates_imag = torch.cat([torch.zeros_like(gates_imag[..., :1]), gates_imag], dim=-1)[..., :-1].contiguous()
        else:
            padded_shifted_gates_real = torch.cat([gates_real, torch.ones_like(gates_real[..., :1])], dim=-1)[..., 1:].contiguous()
            padded_shifted_gates_imag = torch.cat([gates_imag, torch.zeros_like(gates_imag[:, :, :1])], dim=-1)[:, :, 1:].contiguous()

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
            enable_fp_fusion=False
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
            # For x_t = a_t x_{t+1} + b_t → ∂L/∂a_t = ∂L/∂x_t * x_{t+1}
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
    
    @staticmethod
    def forward(gates, tokens, reverse=False):
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
        #print(gates_real.data_ptr(), gates_imag.data_ptr(), tokens_real.data_ptr())
        forward_scan_btc_complex[(B, C)](
            gates_real, gates_imag,
            tokens_real, tokens_imag,
            gates_new_real, gates_new_imag,
            tokens_new_real, tokens_new_imag,
            SEQUENCE_LENGTH=T,
            CHANNELS=C,
            REVERSE=reverse,
            enable_fp_fusion=False
        )

        return gates_new, tokens_new

    @staticmethod
    def backward(ctx, grad_gates, grad_tokens):
        # Given a linear scan: x_{t+1} = g_{t+1} * x_t + b_{t+1}
        # dL / dx_t = dL / dx_{t+1} * dx_{t+1}/dx_t = dL / dx_{t+1} * g*_{t+1}
        # grad_gates: dL / dg_t = dL / dx_{t+1} * x*_{t}
        # grad_tokens: dL / db_t = dL / dx_t
    
        reverse = ctx.reverse
        # Padded reverse scan
        gates, tokens,  output_tokens = ctx.saved_tensors
        B, T, C = tokens.shape

        assert grad_tokens.shape == tokens.shape

        # Allocate output tensors
        d_tokens_real = torch.zeros_like(tokens.real, dtype=tokens.real.dtype).contiguous()
        d_tokens_imag = torch.zeros_like(tokens.imag, dtype=tokens.imag.dtype).contiguous()
        d_gates_real = torch.zeros_like(gates.real, dtype=gates.real.dtype).contiguous()
        d_gates_imag = torch.zeros_like(gates.imag, dtype=gates.imag.dtype).contiguous()

        # Split into real and imaginary parts
        gates = gates.conj()
        gates_real, gates_imag = gates.real, gates.imag
        grad_tokens_real, grad_tokens_imag = grad_tokens.real, grad_tokens.imag

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
            enable_fp_fusion=False
        )
        #print('d_gates_real computed', d_gates_real.mT, d_gates_imag.mT)
        #print('d_tokens_real computed', d_tokens_real.mT, d_tokens_imag.mT)

        # grad_gates: dL / dg_t = dL / dx_{t+1} * x*_{t}   
        if reverse:
            padded_outputs = torch.cat(
                [output_tokens[..., 1:, :],
                torch.zeros_like(output_tokens[..., :1, :])], dim=-2)
        else:
            padded_outputs = torch.cat([torch.zeros_like(output_tokens[..., :1, :]), output_tokens], dim=-2)[..., :-1, :]
        d_gates =  torch.complex(d_tokens_real,d_tokens_imag) * padded_outputs.conj()

        # grad_tokens: dL / db_t = dL / dx_{t+1} * g*_{t+1}
        d_tokens = torch.complex(d_tokens_real, d_tokens_imag) # * gates.conj()

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
            # gates: (T, C) - constant for all batches  
            # tokens: (B, T, C) - vectorized over batch dimension
            batch_size = tokens.shape[0]
            gates_expanded = gates.unsqueeze(0).expand_as(tokens).contiguous()
            outputs = ScanBTC.apply(gates_expanded, tokens, reverse)

            # Both outputs are batched
            out_dims = (0, 0)
        elif (0, 0, None):
            assert gates.ndim == 3
            outputs = ScanBTC.apply(gates, tokens, reverse)
            out_dims = (0, 0)
        else:
            raise NotImplementedError(f"Only (None, 0, None) is implemented but requested {in_dims} gates{gates.shape} tokens {tokens.shape}")

        return outputs, out_dims

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # Define the context setup required for functorch transforms
        gates, tokens, reverse = inputs
        _, output_tokens = outputs
        ctx.save_for_backward(gates, tokens, output_tokens)  # Example: saving inputs that may be needed later
        ctx.reverse = reverse
    

def scan_btc_complex(gates, tokens, reverse=False):
    """
    Solve a first-order recurrence relation for complex numbers.
    x_t = a_t x_{t-1} + b_t
    where :math:`a_t` ("gates") and :math:`b_t` ("tokens") are sequences of vectors.

    """
    gates = gates.contiguous()
    tokens = tokens.contiguous()
    return ScanBTC.apply(gates, tokens, reverse)


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
    def forward(gates, tokens, reverse=False):
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
        outputs_gates = torch.zeros_like(gates, dtype=torch.cfloat)
        
        if reverse:
            gates = torch.flip(gates, dims=[-2])
            tokens = torch.flip(tokens, dims=[-2])

        # Initialize (a_prefix, b_prefix) variables
        a_prefix = gates[..., 0, :]
        b_prefix = tokens[..., 0, :]

        # initialize x_0 = a_0 * x_0 + b_prefix,  but we assume x_0 = 0
        outputs[..., 0, :] = tokens[..., 0, :]
        outputs_gates[..., 0, :] = gates[..., 0, :]
    
        for t in range(1, seq_len):
            #outputs[..., t] = gates[..., t] * outputs[..., t-1] + tokens[..., t]

            # (a_prefix, b_prefix) = (a_prefix, b_prefix) o (gates[..., t], tokens[..., t])
            a_prefix, b_prefix = binary_operator((a_prefix, b_prefix), (gates[..., t, :], tokens[..., t, :]))
            # x_t = a_prefix * x_0 + b_prefix
            outputs[..., t, :] = b_prefix
            outputs_gates[..., t, :] = a_prefix
        
        if reverse:
            outputs_gates = torch.flip(outputs_gates, dims=[-2])
            outputs = torch.flip(outputs, dims=[-2])

        return outputs_gates, outputs
        
    @staticmethod
    def backward(ctx, grads_gates, grad_outputs):
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
    def setup_context(ctx, inputs, outputs):
        # Define the context setup required for functorch transforms
        gates, tokens, reverse = inputs
        _, output_tokens = outputs
        ctx.save_for_backward(gates, tokens, output_tokens)  # Example: saving inputs that may be needed later
        ctx.reverse = reverse

    @staticmethod
    def vmap(info, in_dims, gates, tokens, reverse=False):
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
            # gates: (T, C) - constant for all batches
            # tokens: (B, T, C) - vectorized over batch dimension
            batch_size = tokens.shape[0]
            gates_expanded = gates.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            outputs_g, outputs_k = ComplexLinearScanBTC.apply(gates_expanded, tokens, reverse)
            # Return with correct out_dims - both outputs are batched
            out_dims = (0, 0)
        else:
            raise NotImplementedError("Only (None, 0) is implemented")

        return (outputs_g, outputs_k), out_dims


def scan_btc_complex_autograd(gates, tokens, reverse=False):
    gates = gates.contiguous()
    tokens = tokens.contiguous()
    outputs = ComplexLinearScanBTC.apply(gates, tokens, reverse)
    return outputs


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


