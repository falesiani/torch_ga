import torch

import einops
from collections import namedtuple
from typing import Iterable


__all__ = [
    "get_jacobian"
]

Jacobian = namedtuple("Jacobian", ["y", "j"])

from functools import partial


def get_jacobian(fun, x, m=None, slice_in=None, slice_out=None):
    """
    Computes the jacobian of `fun`-tion with respect to the variable `x`.
    Batch evaluates the jacobian with cost of O(1) vs O(d), but has memory cost of O(d) (only work for low d, d <~ 1000).
    Parameters:
    -----------
    fun: callable: function for the jacobian
    x: torch.Tensor : coordinated of the jacobian
    slice_in, slice_out: if not None, will slice the input or output of the Jacobian

    Returns:
    --------
    jac: Jacobian
        Named tuple containing members `y` and `j`.
        `y` is the function value  of `fun` evaluated at `x`
        `j` is the jacobian matrix of `fun` evaluated at `x`
        
    """
    if len(x.shape)==1: x.unsqueeze(0)
    shape = x.shape[:-1]
    d = x.shape[-1]
    x = x.view(-1, d)
    n = x.shape[0]
    z = einops.repeat(x, "n j -> (n i) j", i=d)
    z.requires_grad_(True)
    y = fun(z)    
    if  m is None:
        out_grad = torch.eye(d, device=x.device, dtype=x.dtype).tile(n, 1)
    else:        
        out_grad = torch.zeros(d, m, device=x.device, dtype=x.dtype).tile(n, 1)        
        out_grad[slice_out,slice_in] = 1.
    j = torch.autograd.grad(y, z, out_grad, create_graph=True, retain_graph=True)[0].view(*shape, d, d)

    if not slice_in is None:
        j = j[...,slice_out,slice_in]
    
    return Jacobian(
        y=einops.rearrange(y, "(n i) j -> n i j", i=d)[:, 0, :].view(*shape, -1),
        j=j
    )

