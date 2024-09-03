"""
Compare associative scan functions in PyTorch and JAX.
"""
import torch
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from deep_ssm.models.s5_fjax.jax_func import associative_scan
from deep_ssm.models.s5_fjax.scan_ref import scan as scan_ref
from typing import Tuple


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

@torch.jit.script
def binary_operator_torch(
  q_i: Tuple[torch.Tensor, torch.Tensor], q_j: Tuple[torch.Tensor, torch.Tensor]):
  """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A."""
  A_i, b_i = q_i
  A_j, b_j = q_j
  return A_j * A_i, torch.addcmul(b_j, A_j, b_i)


def jax_scan(gates, tokens):
  """ Wrapper to call jax.lax.associative_scan with vmap applied along axis=0. """

  def scan_fn(gates_, tokens_):
    return jax.lax.associative_scan(binary_operator, (gates_, tokens_))[1]

  return jax.vmap(scan_fn, in_axes=(0, 0))(gates, tokens)


def torch_scan(gates, tokens):
  """ Wrapper to call torch.scan """

  def scan_fn(gates_, tokens_):
    return associative_scan(binary_operator_torch, (gates_, tokens_))[1]

  return torch.vmap(scan_fn, in_dims=(0, 0))(gates, tokens)


class ScanFunctionsTest(absltest.TestCase):

    def setUp(self):
        np.random.seed(0)
        torch.random.manual_seed(0)

    def test_scan_forward_backward(self):
        # Set up input tensors
        batch_size, seqlen, dim = 1, 2, 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        dtype = 'float32'
        torch_dtype = {'float32': torch.float32, 'float64': torch.float64, 'complex64': torch.complex64}[dtype]
        jax_dtype = {'float32': jnp.float32, 'float64': jnp.float64, 'complex64': jnp.complex64}[dtype]

        # Randomly initialize input tensors
        gates = torch.rand(batch_size, seqlen, dim, device=device, requires_grad=True, dtype=torch_dtype)
        tokens = torch.rand(batch_size, seqlen, dim, device=device, requires_grad=True, dtype=torch_dtype)

        # Convert tensors to numpy for JAX
        gates_jax = jnp.array(np.asarray(gates.detach().cpu()), dtype=jax_dtype)
        tokens_jax = jnp.array(np.asarray(tokens.detach().cpu()), dtype=jax_dtype)

        # Verify inputs are equivalent
        self.assertTrue(torch.allclose(gates, torch.tensor(np.asarray(gates_jax), device=device), rtol=1e-5, atol=1e-8),
                        "Gates do not match!")
        #self.assertTrue(torch.allclose(tokens, torch.tensor(np.asarray(tokens_jax), device=device), rtol=1e-5, atol=1e-8),
        #                "Tokens do not match!")

        # Forward Pass
        out_jax = jax_scan(gates_jax, tokens_jax)
        out_jax_torch = torch.tensor(np.asarray(out_jax), device=device)  # Convert back to torch for comparison
        out_torch_v0 = torch_scan(gates, tokens)
        out_torch_v1 = scan_ref(gates, tokens)

        # Compare Forward Pass
        self.assertTrue(torch.allclose(out_torch_v0, out_jax_torch, rtol=1e-5, atol=1e-8),
                        "Forward pass v0 and jax do not match!")
        self.assertTrue(torch.allclose(out_torch_v1, out_jax_torch, rtol=1e-5, atol=1e-8),
                        "Forward pass v1 and jax do not match!")

        # Backward Pass for out_torch_v1
        loss_torch1 = out_torch_v1.sum()
        loss_torch1.backward(retain_graph=True)
        torch_gates_grad_v1 = gates.grad.clone()
        torch_tokens_grad_v1 = tokens.grad.clone()

        # Reset gradients before the next backward pass
        gates.grad.zero_()
        tokens.grad.zero_()

        # Backward Pass for out_torch_v0
        loss_torch0 = out_torch_v0.sum()
        loss_torch0.backward(retain_graph=True)
        torch_gates_grad_v0 = gates.grad.clone()
        torch_tokens_grad_v0 = tokens.grad.clone()

        # Reset gradients before the JAX comparison
        gates.grad.zero_()
        tokens.grad.zero_()

        def jax_forward(g, t):
            return jax_scan(g, t).sum()

        jax_grads = jax.grad(jax_forward)(gates_jax, tokens_jax)
        # Backward Pass in JAX
        jax_gates_grad = torch.tensor(np.asarray(jax_grads[0]), device=device)
        jax_tokens_grad = torch.tensor(np.asarray(jax_grads[1]), device=device)

        # Compare Backward Pass (Gradients)
        self.assertTrue(torch.allclose(torch_gates_grad_v1, jax_gates_grad, rtol=1e-5, atol=1e-8),
                        "Gates gradients v1 and jax do not match!")
        #self.assertTrue(torch.allclose(torch_tokens_grad_v1, jax_tokens_grad, rtol=1e-5, atol=1e-8),
        #                "Tokens gradients v1 and jax do not match!")
        self.assertTrue(torch.allclose(torch_gates_grad_v0, jax_gates_grad, rtol=1e-5, atol=1e-8),
                        "Gates gradients v0 and jax do not match!")
        #self.assertTrue(torch.allclose(torch_tokens_grad_v0, jax_tokens_grad, rtol=1e-5, atol=1e-8),
        #                "Tokens gradients v0 and jax do not match!")

if __name__ == "__main__":
    absltest.main()
