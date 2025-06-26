import torch
import triton
import triton.language as tl
import os

os.environ["TRITON_INTERPRET"]="1"
os.environ["TRITON_LOG_LEVEL"] = "debug"


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
    strides = base_offset + tl.arange(0, SEQUENCE_LENGTH)

    # we are dealing with complex numbers with real and imag parts interleaved in memory
    strides = strides * 2

    # Return a tensor whose values are loaded from memory at located specified by the pointer
    tokens_r = tl.load(tokens_real + strides)
    tokens_i = tl.load(tokens_imag + strides)
    gates_r = tl.load(gates_real + strides)
    gates_i = tl.load(gates_imag + strides)

    #print("fwd in triton gates_r, gates_i, tokens_r, tokens_i:")
    #print(gates_r, gates_i, tokens_r, tokens_i)
    # Perform scan operation
    tokens_new_r, tokens_new_i, gates_r_out, gates_i_out = tl.associative_scan(
        (tokens_r, tokens_i, gates_r, gates_i),
        axis=0,
        combine_fn=first_order_op_complex
    )

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
    CHANNELS: tl.constexpr,
    ):
    """Backward scan with direct complex number handling."""
    # Program IDs for Batch and Channel (dimensions along which we parallelize)
    batch_id = tl.program_id(axis=0)  # Batch index
    channel_id = tl.program_id(axis=1)  # Channel index

    base_offset = batch_id * CHANNELS * SEQUENCE_LENGTH + channel_id * SEQUENCE_LENGTH

    # Compute reversed strides for the sequence dimension
    reversed_strides = base_offset + (SEQUENCE_LENGTH - 1 - tl.arange(0, SEQUENCE_LENGTH))

    # Load data in reverse order
    gates_r = tl.load(gates_real + reversed_strides)
    gates_i = tl.load(gates_imag + reversed_strides)
    grad_r = tl.load(grad_real + reversed_strides *2)
    grad_i = tl.load(grad_imag + reversed_strides *2)

    #print("bwd in triton gates_r, gates_i, grad_r, grad_i:")
    #print(gates_r, gates_i, grad_r, grad_i)
    # Perform backward scan
    tokens_new_r, tokens_new_i, gates_r_out, gates_i_out = tl.associative_scan(
        (grad_r, grad_i, gates_r, gates_i),
        axis=0,
        combine_fn=first_order_op_complex
    )
    # Debug intermediate results
    #print("tokens_new_r, tokens_new_i, gates_r_out, gates_i_out:")
    #print(tokens_new_r, tokens_new_i, gates_r_out, gates_i_out)

    # Store results in reverse order
    tl.store(d_tokens_real + reversed_strides, tokens_new_r)
    tl.store(d_tokens_imag + reversed_strides, tokens_new_i)
    tl.store(d_gates_real + reversed_strides, gates_r_out)
    tl.store(d_gates_imag + reversed_strides, gates_i_out)


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

        # Backward scan
        backward_scan_bct_complex[(B, C)](
            padded_shifted_gates_real, padded_shifted_gates_imag,
            grad_tokens_real, grad_tokens_imag,
            d_gates_real, d_gates_imag,
            d_tokens_real, d_tokens_imag,
            SEQUENCE_LENGTH=T,
            CHANNELS=C,
            enable_fp_fusion=False
        )
        #print('d_tokens_real computed', d_tokens_real)
        #print('d_gates_real computed', d_gates_real)
        # grad_gates: dL / dg_t = dL / dx_{t+1} * x*_{t}
       
        padded_outputs = torch.cat([torch.zeros_like(output_tokens[..., :1]), output_tokens], dim=-1)[..., :-1]
        d_gates =  torch.complex(d_tokens_real,d_tokens_imag) * padded_outputs.conj()

        # grad_tokens: dL / db_t = dL / dx_{t+1} * g*_{t+1}
        d_tokens = torch.complex(d_tokens_real, d_tokens_imag) # * gates.conj()

        return d_gates, d_tokens

    @staticmethod
    def vmap(info, in_dims, gates, tokens):
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
            # gates: (C, T) - constant across batch
            # tokens: (B, C, T) - vectorized over batch dimension
            B = tokens.shape[0]
            # Expand gates to match batch dimension
            gates_expanded = gates.unsqueeze(0).expand(B, -1, -1).contiguous()
            outputs_g, outputs_k = ScanBCT.apply(gates_expanded, tokens)
            out_dims = (0, 0)  # Both outputs vectorized over batch dimension
        else:
            raise NotImplementedError("Only (None, 0) is implemented")

        return (outputs_g, outputs_k), out_dims

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # Define the context setup required for functorch transforms
        gates, tokens = inputs
        _, output_tokens = outputs
        ctx.save_for_backward(gates, tokens, output_tokens)  # Example: saving inputs that may be needed later
        return ctx  # Return context
        
def scan_bct_complex(gates, tokens):
    """
    Solve a first-order recurrence relation for complex numbers.
    x_t = a_t x_{t-1} + b_t
    where :math:`a_t` ("gates") and :math:`b_t` ("tokens") are sequences of vectors.

    """
    gates = gates.contiguous()
    tokens = tokens.contiguous()
    return ScanBCT.apply(gates, tokens)


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
        forward_scan_btc_complex[(B, C)](
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

        assert grad_tokens.shape == tokens.shape

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
        backward_scan_btc_complex[(B, C)](
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
    def vmap(info, in_dims, gates, tokens):
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
            batch_size = tokens.shape[0]
            gates_expanded = gates.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            outputs = ScanBTC.apply(gates_expanded, tokens)

            # Both outputs are batched
            out_dims = (0, 0)
        else:
            raise NotImplementedError("Only (None, 0) is implemented")

        return outputs, out_dims

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # Define the context setup required for functorch transforms
        gates, tokens = inputs
        _, output_tokens = outputs
        ctx.save_for_backward(gates, tokens, output_tokens)  # Example: saving inputs that may be needed later
        return ctx  # Return context
        
    

def scan_btc_complex(gates, tokens):
    """
    Solve a first-order recurrence relation for complex numbers.
    x_t = a_t x_{t-1} + b_t
    where :math:`a_t` ("gates") and :math:`b_t` ("tokens") are sequences of vectors.

    """
    gates = gates.contiguous()
    tokens = tokens.contiguous()
    return ScanBTC.apply(gates, tokens)


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

