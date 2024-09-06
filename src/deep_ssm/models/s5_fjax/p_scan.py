"""
From https://github.com/sustcsonglin/mamba-triton
"""
import math

import torch
# credit: https://github.com/alxndrTL/mamba.py/blob/main/pscan.py
from einops import einsum, rearrange, repeat
from typing import Tuple
"""

An implementation of the parallel scan operation in PyTorch (Blelloch version).
This code follows the skeleton proposed by Francois Fleuret in his pscan. However, the keys differences are :
-it has been written in an iterative way (rather than recursive)
-the backward pass has been rewritten

Please see docs/pscan.ipynb for a detailed explanation of what happens here.

"""


#  TODO eviter les .flip() en codant un pscan reverse (avec flag)
def flip_complex(tensor:torch.Tensor, dim: int) -> torch.Tensor:
    real_part = tensor.real.flip(dim)
    imag_part = tensor.imag.flip(dim)
    return torch.complex(real_part, imag_part)


class PScan(torch.autograd.Function):
  # automatically generate a vmap rule
  # https://pytorch.org/docs/master/notes/extending.func.html
  generate_vmap_rule = True

  def __init__(self):
    super().__init__()

  @staticmethod
  def setup_context(ctx, inputs, outputs):
    """
    This method prepares the context `ctx` for using functorch transforms.
    """
    # Store inputs for the backward pass
    a, x = inputs
    ctx.save_for_backward(a, x)
  
  @staticmethod
  def pscan(A, X):
    # Convert complex tensors to real and imaginary parts
    # A_real_imag and X_real_imag will have shape (D, T, 2)
    #A = A.contiguous()
    #X = X.contiguous()

    A_real_imag = torch.view_as_real(A)  # Converts complex to real with an extra dimension
    X_real_imag = torch.view_as_real(X)

    D, T, _ = A_real_imag.size()
    num_steps = int(math.log2(T))

    # Up-sweep (reduction) step
    for k in range(num_steps):
        T = 2 * (X_real_imag.size(1) // 2)
        A_real_imag = A_real_imag[:, :T].view(D, T // 2, 2, -1)  # New shape: (D, T//2, 2 (pairs), 2 (real/imag))
        X_real_imag = X_real_imag[:, :T].view(D, T // 2, 2, -1)

        # Update X and A based on real and imaginary parts separately
        X_real_imag[:, :, 1].add_(
            A_real_imag[:, :, 1].mul(X_real_imag[:, :, 0])
        )
        A_real_imag[:, :, 1].mul_(A_real_imag[:, :, 0])

        A_real_imag = A_real_imag[:, :, 1]  # Update A to next level
        X_real_imag = X_real_imag[:, :, 1]  # Update X to next level

    # Down-sweep step
    for k in range(num_steps - 1, -1, -1):
        A_real_imag = torch.view_as_real(A[:, 2 ** k - 1:T:2 ** k])
        X_real_imag = torch.view_as_real(X[:, 2 ** k - 1:T:2 ** k])

        T = 2 * (X_real_imag.size(1) // 2)
        if T < X_real_imag.size(1):
            X_real_imag[:, -1].add_(A_real_imag[:, -1].mul(X_real_imag[:, -2]))
            A_real_imag[:, -1].mul_(A_real_imag[:, -2])

        A_real_imag = A_real_imag[:, :T].view(D, T // 2, 2, 2)
        X_real_imag = X_real_imag[:, :T].view(D, T // 2, 2, 2)

        X_real_imag[:, 1:, 0].add_(A_real_imag[:, 1:, 0].mul(X_real_imag[:, :-1, 1]))
        A_real_imag[:, 1:, 0].mul_(A_real_imag[:, :-1, 1])

    # After completing the operation, convert back to complex
    A = torch.view_as_complex(A_real_imag)
    X = torch.view_as_complex(X_real_imag)

  """
  @staticmethod
  def pscan(A, X):
    #  A : (D, L)
    #  X : (D, L)

    #  modifies X in place by doing a parallel scan.
    #  more formally, X will be populated by these values :
    #  H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
    #  which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

    D, L = A.size()
    num_steps = int(math.log2(L))

    #  up sweep or reduction step
    Aa = A
    Xa = X
    for k in range(num_steps):
      T = 2 * (Xa.size(1) // 2)
      # original A
      Aa = Aa[:, :T].view(D, T // 2, 2)
      Xa = Xa[:, :T].view(D, T // 2, 2)

      Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
      Aa[:, :, 1].mul_(Aa[:, :, 0])

      Aa = Aa[:, :, 1]
      Xa = Xa[:, :, 1]

    #  down sweep
    for k in range(num_steps - 1, -1, -1):
      Aa = A[:, 2 ** k - 1:L:2 ** k]
      Xa = X[:, 2 ** k - 1:L:2 ** k]

      T = 2 * (Xa.size(1) // 2)

      if T < Xa.size(1):
        Xa[:, -1].add_(Aa[:, -1].mul(Xa[:, -2]))
        Aa[:, -1].mul_(Aa[:, -2])

      Aa = Aa[:, :T].view(D, T // 2, 2)
      Xa = Xa[:, :T].view(D, T // 2, 2)

      Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
      Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
  """
  @staticmethod
  def forward(A_in, X_in):
    """
    Applies the parallel scan operation, as defined above. Returns a new tensor.

    Args:
        A_in : (L, D)
        X_in : (L, D)

    Returns:
        A: (L, D)
        H : (L, D) in place changes
    """
    #  clone tensor (in-place ops)
    A = A_in.clone()  #  (L, D)
    X = X_in.clone()  # (L, D)

    # prepare tensors
    A = A.transpose(1, 0)  #  (D, L)
    X = X.transpose(1, 0)  #  (D, L)


    #  parallel scan
    PScan.pscan(A, X) #  (D, L)

    return X.transpose(1, 0)  #  (L, D)
  
  @staticmethod
  def backward(ctx, grad_output_in):
    """
    Flows the gradient from the output to the input. Returns two new tensors.

    Args:
        ctx : A_in : ( L, D), X : (D, L)
        grad_output_in : (L, D)

    Returns:
        gradA : (L, D),
        gradX : (L, D)
    """
    #reverse = ctx.reverse  # Access the reverse flag from the context
    A_in, X = ctx.saved_tensors

    #  clone tensors
    A = A_in.clone()
    #  grad_output_in will be cloned with flip()

    # prepare tensors
    A = A.transpose(1, 0)  #  (D, L)
    A = torch.cat((A[:, :1], flip_complex(A[:, 1:], 1)), dim=1)
    grad_output_b = grad_output_in.transpose(1, 0)

    #  reverse parallel scan
    grad_output_b = flip_complex(grad_output_b, 1)
    PScan.pscan(A, grad_output_b)
    grad_output_b = flip_complex(grad_output_b, 1)

    # Does it work equally for different values
    # TODO: check flip
    grad_output_b =grad_output_b.transpose(1,0)
    Q = torch.zeros_like(X)   # L x D
    Q[:, 1:].add_(X[:, :-1] * grad_output_b[:, 1:])


    return Q, grad_output_b


pscan_ = PScan.apply
# This function now only forwards A_in and X_in, and handles reverse inside

def pscan(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  x = x.contiguous()
  y = y.contiguous()
  out_x = pscan_(x, y)
  return  out_x


def pscan_rev(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  x = x.contiguous()
  y = y.contiguous()

  x = flip_complex(x,1)
  y = flip_complex(y,1)

  out_x = pscan_(x, y)

  out_x = flip_complex(out_x,1)
  return  out_x


