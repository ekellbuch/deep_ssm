"""
    Compare the forward and backward pass of an associative scan for linear recurrence using gates and tokens.

    Example usage for test v0:
    x_t = a_t x_{t-1} + b_t
    
    a_t = 1, 1, 1, 1
    b_t = 0, 1, 2, 3
    
    Forward pass:
    x_0 = 0
    x_1 = 1*x_0 + 0 = 0
    x_2 = 1*x_1 + 1 = 1
    x_3 = 1*x_2 + 2 = 3
    x_4 = 1*x_3 + 3 = 6

    f = sum_t (abs(x_t)) = 0 + 0 + 1 + 3 + 6 = 10
    df/dx_t = sign(x_t)

    Backward pass:

    Gradient w.r.t. b_t
    df/db_t = df/dx_t * dx_t/db_t  = df/dx_t 
        where dx_t/db_t = 1

    df/db_1 = df/dx_1 = 0
    df/db_2 = df/dx_2 = 1
    df/db_3 = df/dx_3 = 1
    df/db_4 = df/dx_4 = 1

    Gradient w.r.t. a_t
    #because a_t affects x_t and all subsequent x_{t+1}, the gradient accumulates
    backward through the chain rule

    df/da_4 = d(|x_4| + |x_3| + |x_2| + |x_1|) 
    where x_4 = a_4 * x_3 + b_4
    df/dx_4 = df/dx_4 * dx_4/da_4 = sign(x_4) * x3 = 1 * 3 = 3
    df/da_3 = df/dx_3 * dx_3/da_3 + df/dx_4 * dx_4/dx_3 * dx_3/da_3 = 1 * 1 + 1 * 1 * 1 = 2
    df/da_2 = df/dx_2 * dx_2/da_2 + df/dx_3 * dx_3/dx_2 * dx_2/da_2 + df/dx_4 * dx_4/dx_3 * dx_3/dx_2 * dx_2/da_2 = 0
    df/da_1 = 0 (given that x1 = x0 =0)

"""

import torch
from torch.autograd import Function
import pytest
from timeit import timeit
from deep_ssm.mixers.scan_utils import scan_bct_complex, torch_associative_scanBCT, scan_bct_complex_autograd, scan_bct_complex_packed
from torch.func import vmap
from functools import partial

# Device selection logic
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Metal Performance Shaders on macOS
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # CUDA on NVIDIA GPUs
else:
    DEVICE = torch.device("cpu")  # Fallback to CPU



#"""
only_reals =[False]
complexities = ["v0","v1"]
batch_dims = [1,2]
feature_dims = [1,2]
seq_lens = [4]
reverses = [False,True]

"""

only_reals =[False]
complexities = ["v1"]
batch_dims = [2]
feature_dims = [2]
seq_lens = [4]
reverses = [False,True]
"""

def create_inputs(batch, dim, seq_len, complexity, only_real):
    if complexity == "v0":
        """
        x_t = a_t * x_{t-1} + b_t
        a_t = [1, 1, 1, 1]
        b_t = [0, 1, 2, 3]/4
        """
        gates = torch.ones(batch, dim, seq_len, dtype=torch.cfloat)
        tokens = torch.arange(batch*dim*seq_len).view(batch, dim, seq_len).type(torch.cfloat)/(batch*dim*seq_len)
    elif complexity == "v1":
        """
        x_t = a_t * x_{t-1} + b_t
        a_t = [0, 1, 2, 3]/4
        b_t = [0, 1, 2, 3]/4
        """
        gates = torch.arange(batch*dim*seq_len).view(batch, dim, seq_len).type(torch.cfloat)/(batch*dim*seq_len)
        tokens = torch.arange(batch*dim*seq_len).view(batch,dim, seq_len).type(torch.cfloat)/(batch*dim*seq_len)
    elif complexity == "v2":
        """
        x_t = a_t * x_{t-1} + b_t
        a_t = [1, 1, 1, 1]
        b_t is random
        """
        gates = torch.ones(batch, dim, seq_len, dtype=torch.cfloat)
        tokens = torch.randn(batch, dim, seq_len, dtype=torch.cfloat)
    elif complexity == "v3":
        gates = torch.randn(batch*dim*seq_len).view(batch, dim, seq_len).type(torch.cfloat)
        tokens = torch.ones(batch*dim*seq_len).view(batch, dim,seq_len).type(torch.cfloat)
    elif complexity == "v4":
        gates = torch.randn(batch, dim, seq_len, dtype=torch.cfloat)
        tokens = torch.randn(batch, dim, seq_len, dtype=torch.cfloat)

    if only_real:
        gates.imag = 0 
        tokens.imag = 0

    gates = gates.to(DEVICE)
    tokens = tokens.to(DEVICE)
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
@pytest.mark.parametrize("reverse", reverses)
def test_scan_functions(only_real, complexity, batch, dim, seq_len, reverse):
    # For a linear scan: x_{t+1} = g_{t+1} * x_t + b_{t+1}
    # compare a vanilla pytorch implementation, a custom autograd implementation, and
    torch.manual_seed(42)

    gates, tokens = create_inputs(batch, dim, seq_len, complexity, only_real)

    # ----------------------------
    # PyTorch Sequential Implementation
    tokens.grad = None  # Reset gradients
    gates.grad = None
    out_g, out_token = torch_associative_scanBCT(gates, tokens, reverse)

    # Compute a loss
    loss_seq = out_token.abs().sum()
    loss_seq.backward()

    seq_gates_grad = gates.grad.clone()
    seq_tokens_grad = tokens.grad.clone()

    # ----------------------------
    # CustomAutograd Implementation
    tokens.grad = None  # Reset gradients
    gates.grad = None

    outputs_g, outputs_t = scan_bct_complex_autograd(gates, tokens, reverse)

    # Compute a loss
    loss = outputs_t.abs().sum()
    loss.backward()

    seq2_gates_grad = gates.grad.clone()
    seq2_tokens_grad = tokens.grad.clone()
    
    # Check forward pass match
    atol = 1e-5
    assert torch.allclose(out_token, outputs_t, atol=atol), print(f"Forward pass mismatch token vanilla {out_token} and torch {outputs_t}" )

    # Check backward pass match
    assert torch.allclose(seq_tokens_grad, seq2_tokens_grad, atol=atol), print(f"Backward pass mismatch token vanilla {seq_tokens_grad} and torch {seq2_tokens_grad}" )
    assert torch.allclose(seq_gates_grad, seq2_gates_grad, atol=atol), print(f"Backward pass mismatch gates vanilla {seq_gates_grad} and torch {seq2_gates_grad}" )

    if not out_token.is_cuda:
        pytest.skip("CUDA not available, skipping triton test")
    # ----------------------------
    # Triton Implementation
    tokens.grad = None  # Reset gradients
    gates.grad = None

    new_g, new_t = scan_bct_complex(gates, tokens, reverse)

    # Compute a loss
    loss_tri = new_t.abs().sum()
    loss_tri.backward()

    tri_gates_grad = gates.grad.clone()
    tri_tokens_grad = tokens.grad.clone()

    # Check forward pass match
    atol = 1e-5
    assert torch.allclose(out_token, new_t, atol=atol),  print(f"Forward pass mismatch token vanilla {out_token} and triton {new_t}" )

    # Check backward pass match
    assert torch.allclose(seq_tokens_grad, tri_tokens_grad, atol=atol), print(f"Backward pass mismatch token vanilla {seq_tokens_grad} and triton {tri_tokens_grad}" )
    assert torch.allclose(seq_gates_grad, tri_gates_grad, atol=atol), print(f"Backward pass mismatch gates  vanilla {seq_gates_grad} and triton {tri_gates_grad}" )

    # ----------------------------
    # Triton Implementation packed
    tokens.grad = None  # Reset gradients
    gates.grad = None
    new_g, new_t = scan_bct_complex_packed(gates, tokens, reverse)

    # Compute a loss
    loss_tri = new_t.abs().sum()
    loss_tri.backward()

    tri_gates_grad = gates.grad.clone()
    tri_tokens_grad = tokens.grad.clone()
    # Check forward pass match
    atol = 1e-5

    assert torch.allclose(out_token, new_t, atol=atol),  print(f"Forward pass mismatch token vanilla {out_token} and triton packed {new_t}" )
    # Check backward pass match
    assert torch.allclose(seq_tokens_grad, tri_tokens_grad, atol=atol), print(f"Backward pass mismatch token vanilla {seq_tokens_grad} and triton packed {tri_tokens_grad}" )
    assert torch.allclose(seq_gates_grad, tri_gates_grad, atol=atol), print(f"Backward pass mismatch gates  vanilla {seq_gates_grad} and triton packed {tri_gates_grad}" )


in_dims_pairs = [(None, 0)]#, (0, 1), (1, 0), (1, 1)]

def test_single_scan_vmap():
    only_real = True
    complexity = "v1"
    batch = 4
    dim = 1
    seq_len = 4
    in_dims = (None, 0)  # vmap over dimension that gets vectorized for each input
    reverse = True
    test_scan_vmap(only_real, complexity, batch, dim, seq_len, in_dims, reverse)


@pytest.mark.parametrize("only_real", only_reals)
@pytest.mark.parametrize("complexity", complexities)
@pytest.mark.parametrize("batch", batch_dims)
@pytest.mark.parametrize("dim", feature_dims)
@pytest.mark.parametrize("seq_len", seq_lens)
@pytest.mark.parametrize("in_dims", in_dims_pairs)
@pytest.mark.parametrize("reverse", reverses)
def test_scan_vmap(only_real, complexity, batch, dim, seq_len, in_dims, reverse):
    """
    vmap: slice over dimension and applies function to each slide:
    Example: (None, 0) 
        First input is treated as constant and the same value is used for all iterations
        Second input is sliced over the specified dimension, function is applied to each slice independently.
    """
    torch.manual_seed(42)
    atol = 1e-5
    gates, tokens = create_inputs(batch, dim, seq_len, complexity, only_real)

    if in_dims[0] is None:
        # gates has shape (C, T) and is constant over batch dimension
        gates = gates[0].detach().requires_grad_()
        out_dims = (0, 0)

    # Let's define a function:
    func_ = lambda x: x.abs().sum((-1,-2)).mean()
    
    # Let's calculate the outputs and grad_outputs structure:
    gates.grad = None
    tokens.grad = None

    outputs_seq_g, outputs_seq_token = torch_associative_scanBCT(gates, tokens, reverse)
    loss = func_(outputs_seq_token)
    loss.backward()
    seq_gates_grad = gates.grad.clone()
    seq_tokens_grad = tokens.grad.clone()

    # Now compare the vmap forward pass:
    gates.grad = None
    tokens.grad = None
    vmap_forward = torch.vmap(lambda g, t: torch_associative_scanBCT(g,t,reverse), in_dims=in_dims, out_dims=out_dims)
    outputs_vmap_g, outputs_vmap_token = vmap_forward(gates, tokens)
    loss = func_(outputs_vmap_token)
    loss.backward()
    vmap_gates_grad = gates.grad.clone()
    vmap_tokens_grad = tokens.grad.clone()

    assert torch.allclose(outputs_seq_g, outputs_vmap_g, atol=atol), print(f"Vmap forward outputs gates {outputs_vmap_g} do not match standard {outputs_seq_g}")
    assert torch.allclose(outputs_seq_token, outputs_vmap_token, atol=atol), print(f"Vmap forward outputs token {outputs_vmap_token} do not match standard {outputs_seq_token}")
    assert torch.allclose(seq_gates_grad, vmap_gates_grad, atol=atol), print(f"Vmap forward gate gradients {vmap_gates_grad} do not match standard {seq_gates_grad}")
    assert torch.allclose(seq_tokens_grad, vmap_tokens_grad, atol=atol), print(f"Vmap forward token gradients {vmap_tokens_grad} do not match standard {seq_tokens_grad}")

    # -------------------------------------
    # Vmap ComplexLinearScanBTC:
    gates.grad = None
    tokens.grad = None
    vmap_forward2 = torch.vmap(lambda g, t: scan_bct_complex_autograd(g, t, reverse=reverse), in_dims=in_dims, out_dims=out_dims)
    outputs2_vmap_g, outputs2_vmap_token = vmap_forward2(gates, tokens)
    loss = func_(outputs2_vmap_token)
    loss.backward()

    # Get the gradients from the standard backward pass
    vmap2_gates_grad = gates.grad.clone()
    vmap2_tokens_grad = tokens.grad.clone()

    assert torch.allclose(outputs_seq_g, outputs2_vmap_g, atol=atol), print(f"Vmap forward 2 outputs gates {outputs2_vmap_g} do not match standard {outputs_seq_g}")
    assert torch.allclose(outputs_seq_token, outputs2_vmap_token, atol=atol), print(f"Vmap forward 2 outputs token {outputs2_vmap_token} do not match standard {outputs_seq_token}")
    assert torch.allclose(seq_gates_grad, vmap2_gates_grad, atol=atol), print(f"Vmap forward 2 gate gradients {vmap2_gates_grad} do not match standard {seq_gates_grad}")
    assert torch.allclose(seq_tokens_grad, vmap2_tokens_grad, atol=atol), print(f"Vmap forward 2 token gradients {vmap2_tokens_grad} do not match standard {seq_tokens_grad}")

    if not outputs_seq_token.is_cuda:
        pytest.skip("CUDA not available, skipping triton test")

    # -------------------------------------
    # Vmap triton scan_BTC_complex:
    gates.grad = None
    tokens.grad = None
    vmap_forward3 = torch.vmap(lambda g, t: scan_bct_complex(g,t, reverse), in_dims=in_dims, out_dims=out_dims)
    outputs3_vmap_g, outputs3_vmap_token = vmap_forward3(gates, tokens)
    loss = func_(outputs3_vmap_token)
    loss.backward()

    # Get the gradients from the standard backward pass
    vmap3_gates_grad = gates.grad.clone()
    vmap3_tokens_grad = tokens.grad.clone()

    assert torch.allclose(outputs_seq_g, outputs3_vmap_g, atol=atol), print(f"Vmap forward 3 outputs gates {outputs3_vmap_g} do not match standard {outputs_seq_g}")
    assert torch.allclose(outputs_seq_token, outputs3_vmap_token, atol=atol), print(f"Vmap forward 3 outputs token {outputs3_vmap_token} do not match standard {outputs_seq_token}")
    assert torch.allclose(seq_gates_grad, vmap3_gates_grad, atol=atol), print(f"Vmap forward 3 gate gradients {vmap3_gates_grad} do not match standard {seq_gates_grad}")
    assert torch.allclose(seq_tokens_grad, vmap3_tokens_grad, atol=atol), print(f"Vmap forward 3 token gradients {vmap3_tokens_grad} do not match standard {seq_tokens_grad}")

    # -------------------------------------
    # Vmap triton scan_BTC_complex:
    gates.grad = None
    tokens.grad = None
    vmap_forward3 = torch.vmap(lambda g, t: scan_bct_complex_packed(g,t, reverse), in_dims=in_dims, out_dims=out_dims)
    outputs3_vmap_g, outputs3_vmap_token = vmap_forward3(gates, tokens)
    loss = func_(outputs3_vmap_token)
    loss.backward()

    # Get the gradients from the standard backward pass
    vmap3_gates_grad = gates.grad.clone()
    vmap3_tokens_grad = tokens.grad.clone()

    assert torch.allclose(outputs_seq_g, outputs3_vmap_g, atol=atol), print(f"Vmap forward 3 outputs gates {outputs3_vmap_g} do not match standard {outputs_seq_g}")
    assert torch.allclose(outputs_seq_token, outputs3_vmap_token, atol=atol), print(f"Vmap forward 3 outputs token {outputs3_vmap_token} do not match standard {outputs_seq_token}")
    assert torch.allclose(seq_gates_grad, vmap3_gates_grad, atol=atol), print(f"Vmap forward 3 gate gradients {vmap3_gates_grad} do not match standard {seq_gates_grad}")
    assert torch.allclose(seq_tokens_grad, vmap3_tokens_grad, atol=atol), print(f"Vmap forward 3 token gradients {vmap3_tokens_grad} do not match standard {seq_tokens_grad}")

    return