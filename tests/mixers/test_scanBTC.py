import torch
from torch.autograd import Function
import os
import pytest
from torch.func import vmap, vjp
from torch.utils._pytree import tree_flatten, tree_unflatten

from deep_ssm.mixers.scan_utils import scan_btc_complex, torch_associative_scanBTC, scan_btc_complex_autograd


# -------------------------------------------------------------------------------------------------
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

# -------------------------------------------------------------------------------------------------

only_reals =[False]
complexities = ["v0", "v4"]
batch_dims = [1, 4]
feature_dims = [1, 4]
seq_lens = [4, 256]
reverses = [False,True]

"""
only_reals =[True]
complexities = ["v0"]#, "v1", "v2", "v3", "v4"]
batch_dims = [2]
feature_dims = [1]
seq_lens = [4]

#"""

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
    out_g, out_token = torch_associative_scanBTC(gates, tokens, reverse)

    # Compute a loss
    loss_seq = out_token.abs().sum()
    loss_seq.backward()

    seq_gates_grad = gates.grad.clone()
    seq_tokens_grad = tokens.grad.clone()

    # ----------------------------
    # CustomAutograd Implementation
    tokens.grad = None  # Reset gradients
    gates.grad = None

    outputs_g, outputs_t = scan_btc_complex_autograd(gates,tokens, reverse)

    # Compute a loss
    loss = outputs_t.abs().sum()
    loss.backward()

    seq2_gates_grad = gates.grad.clone()
    seq2_tokens_grad = tokens.grad.clone()

    # ----------------------------
    # Triton Implementation
    tokens.grad = None  # Reset gradients
    gates.grad = None

    new_g, new_t = scan_btc_complex(gates, tokens, reverse)

    # Compute a loss
    loss_tri = new_t.abs().sum()
    loss_tri.backward()

    tri_gates_grad = gates.grad.clone()
    tri_tokens_grad = tokens.grad.clone()

    # Check forward pass match
    assert torch.allclose(out_token, outputs_t, atol=1e-5), print(f"Forward pass mismatch vanilla {out_token} and custom {outputs_t}" )
    assert torch.allclose(out_token, new_t, atol=1e-5),  print(f"Forward pass mismatch vanilla {out_token} and triton {new_t}" )

    # Check backward pass match
    assert torch.allclose(seq_tokens_grad, seq2_tokens_grad, atol=1e-5), print(f"Backward pass mismatch token vanilla {seq_tokens_grad} and custom {seq2_tokens_grad}" )
    assert torch.allclose(seq_tokens_grad, tri_tokens_grad, atol=1e-5), print(f"Backward pass mismatch token vanilla {seq_tokens_grad} and triton {tri_tokens_grad}" )
    assert torch.allclose(seq_gates_grad, seq2_gates_grad, atol=1e-5), print(f"Backward pass mismatch gates vanilla {seq_gates_grad} and custom {seq2_gates_grad}" )
    assert torch.allclose(seq_gates_grad, tri_gates_grad, atol=1e-5), print(f"Backward pass mismatch gates  vanilla {seq_gates_grad} and triton {tri_gates_grad}" )


def single_backwardBTC_with_vjp(gates, tokens, grad_gates, grad_tokens):
    """Backward pass using vjp.
    Call vjp from Pytorch to compute (single forward BTC pass)
    vjp_fn is a function that computes the vector-Jacobian product.

    vjp_fn takes in the gradients of the loss with respect to the inputs.

    """
    outputs, vjp_fn = vjp(scan_btc_complex_autograd, gates, tokens)
    grads = vjp_fn((grad_gates, grad_tokens))
    return grads


def tri_backwardBTC_with_vjp(gates, tokens, grad_gates, grad_tokens):
    """Backward pass using vjp."""
    outputs, vjp_fn = vjp(scan_btc_complex, gates, tokens)
    grads = vjp_fn((grad_gates, grad_tokens))
    assert len(grads) == 2, f"Expected 2 gradients, got {len(grads)}"
    return grads

in_dims_pairs = [(None, 0)]#, (0, 1), (1, 0), (1, 1)]

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
        # For (None, 0) case: gates should be (T, C), tokens should be (B, T, C)
        gates = gates[0].detach().contiguous().requires_grad_()
        out_dims = (0, 0)

    # Let's define a function:
    func_ = lambda x: x.abs().sum((-1,-2)).mean()
    
    # Let's calculate the outputs and grad_outputs structure:
    gates.grad = None
    tokens.grad = None

    outputs_seq_g, outputs_seq_token = torch_associative_scanBTC(gates, tokens, reverse)
    loss = func_(outputs_seq_token)
    loss.backward()
    seq_gates_grad = gates.grad.clone()
    seq_tokens_grad = tokens.grad.clone()

    # Now compare the vmap forward pass:
    gates.grad = None
    tokens.grad = None
    vmap_forward = torch.vmap(lambda g, t: torch_associative_scanBTC(g, t, reverse), in_dims=in_dims, out_dims=out_dims)
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
    vmap_forward2 = torch.vmap(lambda g, t: scan_btc_complex_autograd(g, t, reverse), in_dims=in_dims, out_dims=out_dims)
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

    # -------------------------------------
    # Vmap triton scan_BTC_complex:
    gates.grad = None
    tokens.grad = None
    vmap_forward3 = torch.vmap(lambda g, t: scan_btc_complex(g, t, reverse), in_dims=in_dims, out_dims=out_dims)
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