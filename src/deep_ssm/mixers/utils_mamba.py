import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union
from torch import nn

def causal_conv1d_ref(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """
    This function performs a causal 1D convolution on the input `x`. It optionally takes initial states
    for the convolution, handles convolution with or without padding, and returns the final states if requested.
    The function supports an optional activation function (SiLU or Swish) and processes the input in batches.

    x: (batch, dim, seqlen)
    weight: (dim, width)
       - The weights for the causal 1D convolution, where `width` is the kernel size.

    bias: (dim,)
       - Bias term to be added to the convolution output (if provided).

    initial_states: (batch, dim, width - 1)
        - The initial state of the convolution for each batch and feature dimension. If provided, this state is
         prepended to the input sequence to ensure the convolution uses information from past time steps.

    final_states_out: Optional (batch, dim, width - 1)
       - If provided, the final states are written to this tensor (in-place). If not provided, a new tensor containing
         the final states is returned.

    activation: Optional string
       - Specifies an activation function to apply after the convolution. Can be either "silu" or "swish". If `None`,
         no activation is applied.

    out: (batch, dim, seqlen)
        - A tensor of shape (batch, dim, seqlen) representing the result of the causal 1D convolution.
        If `return_final_states` is `True`, it also returns the final states as a separate tensor.

    """

    # Validate that the activation function is supported (either "silu", "swish", or None)
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    # Store the original data type of the input for conversion at the end
    dtype_in = x.dtype

    # Ensure the input `x` is of the same dtype as the convolution weights
    x = x.to(weight.dtype)

    # Get the sequence length and convolution filter width
    seqlen = x.shape[-1]
    dim, width = weight.shape

    # If no initial states are provided, perform standard causal convolution with padding
    if initial_states is None:
        # Apply 1D convolution with padding to ensure causal behavior (output only depends on current and past values)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        # If initial states are provided, prepend them to the input
        # This allows the convolution to incorporate past information stored in `initial_states`
        x = torch.cat([initial_states, x], dim=-1)

        # Perform 1D convolution without padding since the state has already been prepended
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)

    # The output is truncated to the original sequence length to match the input's time dimension
    out = out[..., :seqlen]

    # If final states are requested, compute and return them
    if return_final_states:
        # Final states are the last `width - 1` values of the convolution input
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in
        )  # (batch, dim, width - 1)

        # If `final_states_out` is provided, copy the final states to the tensor (in-place)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            # If `final_states_out` is not provided, create a new tensor to hold the final states
            final_states_out = final_states

    # Apply the activation function if specified (SiLU is used for both "silu" and "swish")
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)

    # Return the output, and the final states if requested
    return out if not return_final_states else (out, final_states_out)


def causal_conv1d_update_ref(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None):
    """
    This function performs a causal 1D convolution on the input `x` while maintaining and updating
    the convolutional state (`conv_state`). It supports sequence processing where new input data
    is incrementally fed into the model. The function handles both regular state updates and updates
    using a circular buffer when `cache_seqlens` is provided.

    Parameters:
    -----------
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
        - Holds the previous state of the input for performing causal convolution.
        - `state_len`: Length of the convolution state buffer. Must be >= `width - 1` (where `width` is the filter size).

    weight: (dim, width)
       - Convolution filter weights where `width` is the kernel size.

    bias: (dim,)
       - Bias term to be added to the convolution output (if provided).

    activation: Optional
       - Non-linear activation to be applied. Must be either `None`, "silu", or "swish".

    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    Returns:
    --------
    out: (batch, dim) or (batch, dim, seqlen)
        - representing the output of the convolution operation after applying the given activation function (if specified).
    """

    # Validate the activation function
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    # Store the original data type of `x` for conversion at the end
    dtype_in = x.dtype

    # If the input `x` is 2D (batch, dim), add a sequence length dimension
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    # Get dimensions from the input and convolution state
    batch, dim, seqlen = x.shape
    width = weight.shape[1]  # Width of the convolution filter
    state_len = conv_state.shape[-1]  # Length of the convolution state buffer

    # Ensure the shapes of the input and weights are correct
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)

    # If no cache_seqlens is provided, perform a standard causal convolution update
    if cache_seqlens is None:
        # Concatenate the previous state (`conv_state`) with the new input `x`
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)  # (batch, dim, state_len + seqlen)
        # Update the convolution state by keeping the latest `state_len` elements
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        # Handle the convolution state as a circular buffer, based on cache_seqlens

        # Calculate the indices for the filter window based on cache_seqlens
        width_idx = torch.arange(-(width - 1), 0, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)

        # Gather the appropriate past values from conv_state based on width_idx
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)

        # Calculate the indices for updating the convolution state
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)

        # Update the conv_state buffer in a circular manner
        conv_state.scatter_(2, copy_idx, x)

    # Perform the causal 1D convolution using `conv1d` with no padding, followed by the optional bias
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]

    # If the input was 2D, remove the extra sequence dimension
    if unsqueeze:
        out = out.squeeze(-1)

    # Apply the activation function if specified (SiLU is used for both "silu" and "swish")
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    Performs a recursive state-update operation over a sequence of inputs, where the internal state
    evolves at each time step based on the input sequence `u`, scaling factors `delta`, and transformation
    matrices `A`, `B`, and `C`. The function computes an output sequence `y` by combining the evolving
    state with `C` and returns the final output sequence, and optionally the last state.

    The function processes time-dependent data by iterating over the time dimension of `u`, updating the
    state `x` and generating outputs `y` at each step. The state update is performed as:

        x_t = exp(delta_t A) * x_(t-1) + delta_t B u_t

    The output `y_t` at each time step is computed as:

        y_t = C * x_t + D u_t

    Optional features include biasing and softplus activation for `delta`, modulation of the output using
    a `z` tensor, and handling of complex numbers for `A`, `B`, and `C`.

        y_t = y_t * silu(z)

    Parameters:
    -----------
    u: r(B D L)  - input sequence data.
    delta: r(B D L) -  scaling factor applied to the state.
    A: c(D N) or r(D N) - state transformation matrix (possibly complex).
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L) -- state update
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L) -- output updates
    D: r(D) - applied as an output transformation.
    z: r(B D L) - used to modulate the output via SiLU.
    delta_bias: r(D), fp32  - bias added to delta before state update
    delta_softplus: Boolean flag to apply softplus to delta to ensure  positive values.
    return_last_state: Boolean flag to return the last state along with the output.

    Returns:
    --------
    out: r(B D L) ) - the output sequence.
    last_state (optional): r(B D dstate) or c(B D dstate) -- returns last state
    """
    # Store the original data type of u for later conversion
    dtype_in = u.dtype

    # Convert u and delta to float32 for consistent computation
    u = u.float()
    delta = delta.float()

    # If delta_bias is provided, add it to delta
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    # Apply softplus to delta if delta_softplus is True (ensures positive values)
    if delta_softplus:
        delta = F.softplus(delta)
    # Get the batch size, number of dimensions (D), and the internal state size (dstate) from input shapes
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]

    # Determine if B and C are variable (i.e., have more than 2 dimensions)
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    # If A is a complex tensor, convert B and C to complex representations as well
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()

    # Initialize the state `x` as zeros, of shape (B, D, dstate)
    x = A.new_zeros((batch, dim, dstate))

    # Initialize list to store the output sequence at each time step
    ys = []

    # Precompute delta * A for efficiency, performed over the time dimension (L)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))

    # Compute delta * B * u depending on the shape of B
    if not is_variable_B:
        # If B is not variable, use a standard einsum for delta * B * u
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            # If B has 3 dimensions, apply the corresponding einsum
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            # For higher-dimensional B, repeat B over the groups and apply einsum
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)

    # Repeat C for variable-sized inputs if necessary
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])

    # Initialize the variable to store the last state if requested
    last_state = None

    # Loop over the time steps (L) of the input sequence
    for i in range(u.shape[2]):

        # Update the state `x` using the precomputed deltaA and deltaB_u
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]

        # Compute the output y at each time step using C and the state x
        if not is_variable_C:
            # Standard einsum for fixed-size C
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                # Variable-sized C with 3 dimensions
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                # Variable-sized C with 4 dimensions
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])

        # Store the last state if it's the final time step
        if i == u.shape[2] - 1:
            last_state = x
        # If y is a complex tensor, only take the real part and double it
        if y.is_complex():
            y = y.real * 2

        # Append the output y at this time step to the list ys
        ys.append(y)

    # Stack the outputs from all time steps into a tensor of shape (B, D, L)
    y = torch.stack(ys, dim=2) # (batch dim L)

    # If D is provided, apply an additional transformation to the output
    out = y if D is None else y + u * rearrange(D, "d -> d 1")

    # If z is provided, modulate the output using SiLU (sigmoid-linear activation)
    if z is not None:
        out = out * F.silu(z)

    # Convert the output back to the original input data type
    out = out.to(dtype=dtype_in)

    # Return the output, and optionally the last state if requested
    return out if not return_last_state else (out, last_state)


def mamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    """
     This function performs a sequence of operations that are integral to the Mamba model.
     The steps include causal 1D convolution, linear projections, and a state-space model
     scan using the `selective_scan_ref` function. The core idea is to process an input
     signal and a modulation signal (`x` and `z`). The input signal passed through a convolutions,
     compute a series of projections, and use these results in a state-space scan operation. The function
     returns an output sequence after projecting the final scan result back to the desired
     output dimensions.

     Parameters:
     -----------
     xz: Tensor of shape (B, 2*D, L) - A concatenated input and modulation signal.
         - `B`: Batch size.
         - `2*D`: Twice the number of features (since it contains both `x` and `z`).
         - `L`: Sequence length.

     conv1d_weight: Tensor - Weights for the causal 1D convolution.
     conv1d_bias: Tensor - Bias for the causal 1D convolution.

     x_proj_weight: Tensor - Weights for projecting the convolved input.
     delta_proj_weight: Tensor - Weights for projecting the input to the delta term in the state-space model.

     out_proj_weight: Tensor - Weights for projecting the output after the selective scan operation.
     out_proj_bias: Tensor - Bias for the output projection.

     A: Tensor of shape (D, N) or complex variant - The state-space transformation matrix.
         This tensor is used for evolving the state in the scan operation. It can be a real or complex matrix.

     B: Optional Tensor - If provided, this will be used for state updates in the scan. If not provided, it is computed internally.
     C: Optional Tensor - If provided, this will be used for output computations in the scan. If not provided, it is computed internally.
     D: Optional Tensor - If provided, used for an additional transformation in the scan output.

     delta_bias: Optional Tensor - Bias added to the delta values in the state-space scan.

     B_proj_bias: Optional Tensor - Bias added to the `B` projection if computed internally.
     C_proj_bias: Optional Tensor - Bias added to the `C` projection if computed internally.

     delta_softplus: Boolean - If True, applies a softplus function to `delta` to ensure smooth and positive values.

     Returns:
     --------
     A Tensor of shape (B, L, D) representing the processed sequence output.

     Operation Overview:
     -------------------
     1. **Input Splitting**: The input tensor `xz` is split into two parts: `x` and `z`.
        - `x`: The primary input.
        - `z`: The modulation signal.

     2. **Causal 1D Convolution**: The `x` input is passed through a causal 1D convolution using `conv1d_weight` and `conv1d_bias`,
        followed by SiLU activation.
        TODO: improve The convolution ensures the causal nature of the operation.

     3. **Linear Projection**: The result of the convolution is reshaped and projected using `x_proj_weight`, producing an intermediate
        tensor `x_dbl`. This tensor is used to compute several components:
        - `delta`: A time-dependent scaling factor for the state-space scan.
        - `B` and `C`: If not provided, they are computed from `x_dbl`.

     4. **Delta Computation**: A linear projection is applied to the first part of `x_dbl` to compute `delta`. The result is reshaped
        into a tensor of shape `(B, D, L)` to match the expected input format for the selective scan.

     5. **B and C Projections**: If `B` and `C` are not provided as inputs, they are computed from `x_dbl`. These projections are used
        in the state update and output computations of the selective scan operation.

     6. **Selective Scan**: The function calls `selective_scan_ref` to perform a state-space scan using `A`, `B`, `C`, `delta`, and
        other optional parameters like `D` and `z`. The scan processes the sequence step by step, evolving the internal state at
        each time step based on the inputs and producing an output sequence.

     7. **Final Projection**: The output from the selective scan is projected back to the desired output dimensions using a final
        linear projection with `out_proj_weight` and `out_proj_bias`. This step adjusts the output to match the required format.

     The result is a sequence of processed outputs in the shape `(B, L, D)`.
     """
    # Get the length of the sequence (L) from the input
    L = xz.shape[-1]

    # Get the rank of the delta projection matrix and the dimension of the internal state
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)  # Adjust for complex A if necessary

    # Split the input tensor `xz` into two parts: `x` and `z`
    x, z = xz.chunk(2, dim=1)

    # Apply causal 1D convolution to `x` using conv1d_weight and conv1d_bias, with SiLU activation
    #x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")
    x = causal_conv1d_ref(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")

    # Perform linear projection on convolved `x`, reshaping it to ensure the correct layout for downstream operations
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)

    # Compute the delta values using the projected input and delta_proj_weight
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)

    # If `B` is not provided, compute it from the projected input
    if B is None:  # variable B
        # Extract B from the projection (bl d)
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            # Add bias if provided
            B = B + B_proj_bias.to(dtype=B.dtype)

        # reshape B
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()

    # If `C` is not provided, compute it from the projected input
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]   # Extract C from the projection (bl d)
        if C_proj_bias is not None:
            # Add bias if provided
            C = C + C_proj_bias.to(dtype=C.dtype)

        # Reshape C
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()

    # Perform the selective scan operation using the computed or provided `A`, `B`, `C`, 'D' 'z' and delta
    #y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    y = selective_scan_ref(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)

    # Final linear projection on the output of the scan to match the expected output shape (B, L, D)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


def rms_norm_ref(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    dropout_mask=None,
    dropout_mask1=None,
    upcast=False,
):
    """
    This function performs Root Mean Square (RMS) normalization on the input tensor `x`, optionally supporting
    a second parallel normalization (if `x1`, `weight1`, and `bias1` are provided). It also supports optional
    residual connections, dropout, and row-wise scaling.

    RMS normalization normalizes the input based on the Root Mean Square (RMS) of the features in the input.
    This is different from LayerNorm, which subtracts the mean and divides by the standard deviation. RMSNorm
    only divides by the RMS, making it a simplified normalization method.

    Parameters:
    -----------
    x: Tensor
       - The input tensor to be normalized.

    weight: Tensor
       - The weights for scaling after RMS normalization.

    bias: Tensor
       - The bias for shifting after RMS normalization.

    residual: Optional Tensor
       - Residual connection tensor to be added to the input before normalization.

    x1: Optional Tensor
       - A second input tensor for parallel normalization.

    weight1: Optional Tensor
       - Weights for the second normalization (for parallel normalization).

    bias1: Optional Tensor
       - Bias for the second normalization (for parallel normalization).

    eps: float, default=1e-6
       - A small value added to the denominator for numerical stability in normalization.

    dropout_p: float, default=0.0
       - Dropout probability. If greater than 0.0, dropout is applied to the input.

    rowscale: Optional Tensor
       - Row-wise scaling applied to the input before normalization. Should have the same number of rows as `x`.

    prenorm: bool, default=False
       - If `True`, the function will return the pre-normalized inputs as well.

    dropout_mask: Optional Tensor
       - Pre-specified dropout mask to be applied on `x`.

    dropout_mask1: Optional Tensor
       - Pre-specified dropout mask to be applied on `x1`.

    upcast: bool, default=False
       - If `True`, upcasts all inputs to float32 before performing computations and converts back to the original dtype after.

    Returns:
    --------
    - A normalized tensor (or tensors if `x1` is provided).
    - If `prenorm` is `True`, the function also returns the pre-normalized inputs.
    """
    # Preserve original data type for use at the end when converting back to the original dtype
    dtype = x.dtype

    # If upcasting is enabled, cast all relevant inputs to float32 for higher numerical precision
    if upcast:
        x = x.float()
        weight = weight.float()
        bias = bias.float() if bias is not None else None
        residual = residual.float() if residual is not None else residual
        x1 = x1.float() if x1 is not None else None
        weight1 = weight1.float() if weight1 is not None else None
        bias1 = bias1.float() if bias1 is not None else None

    # Ensure row scaling is not used with parallel RMS normalization (i.e., when x1 is provided)
    if x1 is not None:
        assert rowscale is None, "rowscale is not supported with parallel LayerNorm"

    # Apply row-wise scaling if the `rowscale` tensor is provided
    if rowscale is not None:
        x = x * rowscale[..., None]

    # Apply dropout to the input `x` if dropout probability (`dropout_p`) is greater than 0.0
    if dropout_p > 0.0:
        if dropout_mask is not None:
            # Apply dropout using a pre-specified dropout mask (`dropout_mask`)
            x = x.masked_fill(~dropout_mask, 0.0) / (1.0 - dropout_p)
        else:
            # Use standard dropout if no specific mask is provided
            x = F.dropout(x, p=dropout_p)

        # Apply dropout to `x1` if it is provided and a dropout mask or probability is specified
        if x1 is not None:
            if dropout_mask1 is not None:
                x1 = x1.masked_fill(~dropout_mask1, 0.0) / (1.0 - dropout_p)
            else:
                x1 = F.dropout(x1, p=dropout_p)

    # Combine `x` and `x1` if both are provided (this is the parallel normalization case)
    if x1 is not None:
        x = x + x1

    # Add residual connection if `residual` is provided
    if residual is not None:
        x = (x + residual).to(x.dtype)

    # Compute the inverse of the Root Mean Square (RMS) of `x` along the last dimension (features)
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)

    # Normalize `x` using the RMS and scale by `weight`, add bias if provided
    out = ((x * rstd * weight) + bias if bias is not None else (x * rstd * weight)).to(dtype)

    # If no second weight (`weight1`) is provided, return the output (or pre-normalized `x` if `prenorm` is True)
    if weight1 is None:
        return out if not prenorm else (out, x)
    else:
        # If `weight1` is provided, perform a second RMS normalization on `x` (parallel normalization)
        out1 = ((x * rstd * weight1) + bias1 if bias1 is not None else (x * rstd * weight1)).to(
            dtype
        )
        # Return both outputs for parallel RMS normalization, or both with pre-normalized `x` if `prenorm` is True
        return (out, out1) if not prenorm else (out, out1, x)


class RMSNorm_ref(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-5, dropout_p=0.0, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if dropout_p > 0.0:
            self.drop = torch.nn.Dropout(dropout_p)
        else:
            self.drop = None
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_ref(
            x=x,
            weight=self.weight,
            bias=self.bias,
            residual=residual,
            eps=self.eps,
            dropout_p=self.drop.p if self.drop is not None and self.training else 0.0,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


class Block_ref(torch.nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm_ref is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm_ref)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_ref(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm_ref)
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_ref(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm_ref)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def layer_norm_ref(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    dropout_mask=None,
    dropout_mask1=None,
    upcast=False,
):
    """
    This function performs layer normalization on the input tensor `x`, optionally supporting
    a second parallel normalization (if `x1`, `weight1`, and `bias1` are provided). The function
    also handles optional residual connections, dropout, and row-wise scaling.

    Parameters:
    -----------
    x: Tensor
       - The input tensor to be normalized.

    weight: Tensor
       - The weights for scaling after normalization.

    bias: Tensor
       - The bias for shifting after normalization.

    residual: Optional Tensor
       - Residual connection tensor to be added to the input before normalization.

    x1: Optional Tensor
       - A second input tensor for parallel normalization.

    weight1: Optional Tensor
       - Weights for the second layer normalization (for parallel normalization).

    bias1: Optional Tensor
       - Bias for the second layer normalization (for parallel normalization).

    eps: float, default=1e-6
       - A small value added to the denominator for numerical stability in normalization.

    dropout_p: float, default=0.0
       - Dropout probability. If greater than 0.0, dropout is applied to the input.

    rowscale: Optional Tensor
       - Row-wise scaling applied to the input before normalization. Should have the same number of rows as `x`.

    prenorm: bool, default=False
       - If `True`, the function will return the pre-normalized inputs as well.

    dropout_mask: Optional Tensor
       - Pre-specified dropout mask to be applied on `x`.

    dropout_mask1: Optional Tensor
       - Pre-specified dropout mask to be applied on `x1`.

    upcast: bool, default=False
       - If `True`, upcasts all inputs to float32 before performing computations and converts back to the original dtype after.

    Returns:
    --------
    - A normalized tensor (or tensors if `x1` is provided).
    - If `prenorm` is `True`, the function also returns the pre-normalized inputs.
    """
    # Preserve original data type for later use
    dtype = x.dtype

    # If upcasting is enabled, cast all relevant inputs to float32
    if upcast:
        x = x.float()
        weight = weight.float()
        bias = bias.float() if bias is not None else None
        residual = residual.float() if residual is not None else residual
        x1 = x1.float() if x1 is not None else None
        weight1 = weight1.float() if weight1 is not None else None
        bias1 = bias1.float() if bias1 is not None else None

    # Ensure row scaling is not used with parallel LayerNorm (i.e., when x1 is provided)
    if x1 is not None:
        assert rowscale is None, "rowscale is not supported with parallel LayerNorm"

    # Apply row-wise scaling if `rowscale` is provided
    if rowscale is not None:
        x = x * rowscale[..., None]

    # Apply dropout to the input `x` if `dropout_p` > 0.0
    if dropout_p > 0.0:
        if dropout_mask is not None:
            # Apply dropout using a pre-specified mask
            x = x.masked_fill(~dropout_mask, 0.0) / (1.0 - dropout_p)
        else:
            # Use standard dropout if no mask is provided
            x = F.dropout(x, p=dropout_p)
        if x1 is not None:
            if dropout_mask1 is not None:
                x1 = x1.masked_fill(~dropout_mask1, 0.0) / (1.0 - dropout_p)
            else:
                x1 = F.dropout(x1, p=dropout_p)

    # Combine `x` and `x1` if both are provided (parallel LayerNorm case)
    if x1 is not None:
        x = x + x1

    # If a residual connection is provided, add it to the input
    if residual is not None:
        x = (x + residual).to(x.dtype)

    # Perform layer normalization on `x` using `weight` and `bias`
    out = F.layer_norm(x.to(weight.dtype), x.shape[-1:], weight=weight, bias=bias, eps=eps).to(
        dtype
    )

    # If `weight1` is not provided, return `out` (or pre-normalized input `x` if `prenorm` is True)
    if weight1 is None:
        return out if not prenorm else (out, x)
    else:
        # If `weight1` is provided, perform a second layer normalization on `x` (parallel LayerNorm case)
        out1 = F.layer_norm(
            x.to(weight1.dtype), x.shape[-1:], weight=weight1, bias=bias1, eps=eps
        ).to(dtype)
        # Return both outputs for parallel LayerNorm (or pre-normalized `x` if `prenorm` is True)
        return (out, out1) if not prenorm else (out, out1, x)
