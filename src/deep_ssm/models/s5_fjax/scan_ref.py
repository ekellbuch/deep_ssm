import math
import torch
from typing import Tuple


@torch.jit.script
def split(x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    """
    Take a sequence of inputs that represents a tree level,
    and return all left children and all right children.

    Arguments:
        x (torch.Tensor): shape (B, T, C)

    Returns:
        (torch.Tensor, torch.Tensor): shapes (B, T//2, C), (B, T//2, C)

    >>> split(torch.tensor([1,2,3,4,5,6,7,8])[None, :, None])
    (tensor([[[1], [3], [5], [7]]]), tensor([[[2], [4], [6], [8]]]))
    """
    B, T, C = x.size()
    x = x.view(B, T//2, 2, C)
    return x[:, :, 0, :], x[:, :, 1, :]


@torch.jit.script
def merge(lefts: torch.Tensor, rights: torch.Tensor) -> torch.Tensor:
    """
    Take sequences of all left children and sequences of all right children and merge them
    into a single tree level.

    Arguments:
        lefts (torch.Tensor): shape (B, T//2, C)
        rights (torch.Tensor): shape (B, T//2, C)

    Returns:
        (torch.Tensor): shape (B, T, C)

    >>> lefts = torch.tensor([1,3,5,7])[None, :, None]
    >>> rights = torch.tensor([2,4,6,8])[None, :, None]
    >>> merge(lefts, rights)
    tensor([[[1], [2], [3], [4], [5], [6], [7], [8]]])
    """
    B, half, C = lefts.size()
    x = torch.stack([lefts, rights], dim=2)  # (B, T//2, 2, C)
    return x.view(B, half*2, C)


@torch.jit.ignore
def scan1(
    gates: torch.Tensor,
    tokens: torch.Tensor,
    level: int,
    reverse: bool = False
):
    if reverse:
        right_gates, left_gates = split(gates)
        right_x, left_x = split(tokens)
    else:
        left_gates, right_gates = split(gates)
        left_x, right_x = split(tokens)

    # up: sum together
    gates = torch.mul(left_gates, right_gates)
    tokens = torch.add(torch.mul(right_gates, left_x), right_x)

    if level == 1:
        root_gates, root_x = torch.ones_like(tokens), torch.zeros_like(tokens)
    else:
        root_gates, root_x = scan1(gates, tokens, level=level-1, reverse=reverse)

    if reverse:
        # down: right is root, left is left (+) right
        return merge(torch.mul(root_gates, left_gates), root_gates), merge(torch.add(torch.mul(root_x, left_gates), left_x), root_x)
    else:
        # down: left is root, right is left (+) right
        return merge(root_gates, torch.mul(root_gates, left_gates)), merge(root_x, torch.add(torch.mul(root_x, left_gates), left_x))


@torch.jit.ignore
def scan(
    gates: torch.Tensor,
    tokens: torch.Tensor,
    reverse: bool = False
) -> torch.Tensor:
    """Solve a first-order recurrence relation using a reference torch implementation:

    .. math::
        x_t = a_t x_{t-1} + b_t

    where :math:`a_t` ("gates") and :math:`b_t` ("tokens") are sequences of vectors.

    Arguments:
        gates (torch.Tensor): shape (B, T, C), must be contiguous.
        tokens (torch.Tensor): shape (B, T, C), must be contiguous.
        reverse (bool): whether to solve the recurrence in reverse order, defaults to False

    Returns:
        (torch.Tensor): shape (B, T, C)
    """
    B, T, C = tokens.size()
    level = int(torch.log2(torch.tensor(T, dtype=torch.float32)).item())
    _, x = scan1(gates, tokens, level=level, reverse=reverse)
    return torch.add(torch.mul(x, gates), tokens)
