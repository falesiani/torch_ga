"""Blade-related definitions and functions used across the library."""
from enum import Enum
from typing import List, Tuple, Union
import torch


class BladeKind(Enum):
    """Kind of blade depending on its degree."""
    MV = "mv"
    EVEN = "even"
    ODD = "odd"
    SCALAR = "scalar"
    VECTOR = "vector"
    BIVECTOR = "bivector"
    TRIVECTOR = "trivector"
    PSEUDOSCALAR = "pseudoscalar"
    PSEUDOVECTOR = "pseudovector"
    PSEUDOBIVECTOR = "pseudobivector"
    PSEUDOTRIVECTOR = "pseudotrivector"


def get_blade_repr(blade_name: str, _scalar_name="1", _latex_flag=False, base_symbol="e") -> str:
    """Returns the representation to use
    for a given blade.

    Examples:
    - `"12"` -> `"e_12"`
    - `""` -> `"1"`

    Args:
        blade_name: name of the blade in the algebra (eg. `"12"`)

    Returns:
        Representation to use for a given blade
    """
    # if blade_name == "":
    #     return "1"
    # return "e_%s" % blade_name
    if blade_name == "":
        return _scalar_name
    if _latex_flag:
        return f"{base_symbol}_{{{blade_name}}}"
    else:
        return f"{base_symbol}{blade_name}"


def is_blade_kind(blade_degrees: torch.Tensor, kind: List[Union[BladeKind, str]], max_degree: int) -> torch.Tensor:
    """Finds a boolean mask for whether blade degrees are of a given kind.

    Args:
        blade_degrees: list of blade degrees
        kind: kind of blade to check for
        max_degree: maximum blade degree in the algebra

    Returns:
        boolean mask for whether blade degrees are of a given kind
    """
    # Convert kind to string representation
    # for comparison.
    kind = kind.value if isinstance(kind, BladeKind) else kind

    if kind == BladeKind.MV.value:
        # return torch.constant(True, shape=[len(blade_degrees)])
        return torch.full([len(blade_degrees)],True)
    elif kind == BladeKind.EVEN.value:
        return blade_degrees % 2 == 0
    elif kind == BladeKind.ODD.value:
        return blade_degrees % 2 == 1
    elif kind == BladeKind.SCALAR.value:
        return blade_degrees == 0
    elif kind == BladeKind.VECTOR.value:
        return blade_degrees == 1
    elif kind == BladeKind.BIVECTOR.value:
        return blade_degrees == 2
    elif kind == BladeKind.TRIVECTOR.value:
        return blade_degrees == 3
    elif kind == BladeKind.PSEUDOSCALAR.value:
        return blade_degrees == max_degree
    elif kind == BladeKind.PSEUDOVECTOR.value:
        return blade_degrees == max_degree - 1
    elif kind == BladeKind.PSEUDOBIVECTOR.value:
        return blade_degrees == max_degree - 2
    elif kind == BladeKind.PSEUDOTRIVECTOR.value:
        return blade_degrees == max_degree - 3
    raise Exception("Unknown blade kind: %s" % kind)


def set_diff(a,b):
    combined = torch.cat((a.squeeze(), b.squeeze()))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    # intersection = uniques[counts > 1]    
    return difference


def invert_blade_indices(num_blades: int, blade_indices: torch.Tensor) -> torch.Tensor:
    """Returns all blade indices except for the given ones.

    Args:
        num_blades: Total number of blades in the algebra
        blade_indices: blade indices to exclude

    Returns:
        All blade indices except for the given ones
    """

    # all_blades = tf.range(num_blades, dtype=blade_indices.dtype)
    # return tf.sparse.to_dense(tf.sets.difference(
    #     tf.expand_dims(all_blades, axis=0),
    #     tf.expand_dims(blade_indices, axis=0)
    # ))[0]

    all_blades = torch.arange(num_blades, dtype=blade_indices.dtype)
    return set_diff(all_blades.unsqueeze(0), blade_indices.unsqueeze(0))[0]

# pip install functorch
# from functorch import vmap

def get_blade_of_kind_indices(blade_degrees: torch.Tensor, kind: BladeKind,
                              max_degree: int, invert: bool = False) -> torch.Tensor:
    """Finds a boolean mask for whether blades are of a given kind.

    Args:
        blade_degrees: List of blade degrees
        kind: kind of blade for which the mask will be true
        max_degree: maximum blade degree in the algebra
        invert: whether to invert the result

    Returns:
        boolean mask for whether blades are of a given kind
    """
    # cond = is_blade_kind(blade_degrees, kind, max_degree)
    # cond = tf.math.logical_xor(cond, invert)
    # return tf.where(cond)[:, 0]
    
    # cond = is_blade_kind(blade_degrees, kind, max_degree)
    # cond = torch.math.logical_xor(cond, invert)
    # return torch.where(cond)[:, 0]

    # cond = torch.vmap(is_blade_kind(blade_degrees, kind, max_degree))
    # cond = is_blade_kind(blade_degrees, kind, max_degree))
    # cond = cond(invert)
    # return torch.where(cond)[:, 0]

    # print(blade_degrees.shape)
    if False: print("get_blade_of_kind_indices:blade_degrees:",blade_degrees,"kind:",kind)
    cond = is_blade_kind(blade_degrees, kind, max_degree)
    # print("cond:",cond)
    # print(f"cond.shape={cond.shape}")
    cond = torch.logical_xor(cond,invert*torch.ones_like(cond))
    # print(f"cond.shape={cond.shape}")
    # print(f"cond.nonzero().shape={cond.nonzero().shape}")
    # print("cond:",cond)
    # print(cond.shape)
    # return torch.where(cond)[:, 0]
    # return cond[:, 0]
    return cond.nonzero().squeeze()
    # return cond


def _normal_swap(x: List[str]) -> List[str]:
    """Swaps the first unordered blade pair and returns the new list as well
    as whether a swap was performed."""
    for i in range(len(x) - 1):
        a, b = x[i], x[i + 1]
        if a > b:  # string comparison
            x[i], x[i+1] = b, a
            return False, x
    return True, x


def get_normal_ordered(blade_name: str) -> Tuple[int, str]:
    """Returns the normal ordered blade name and its sign.
    Example: 21 => -1, 12

    Args:
        blade_name: Blade name for which to return normal ordered
        name and sign

    Returns:
        sign: sign of the blade
        blade_name: normalized name of the blade
    """
    blade_name = list(blade_name)
    sign = -1
    done = False
    while not done:
        sign *= -1
        done, blade_name = _normal_swap(blade_name)
    return sign, "".join(blade_name)


def get_blade_indices_from_names(blade_names: List[str],
                                 all_blade_names: List[str]) -> torch.Tensor:
    """Finds blade signs and indices for given blade names in a list of blade
    names. Blade names can be unnormalized and their correct sign will be
    returned.

    Args:
        blade_names: Blade names to return indices for. May be unnormalized.
        all_blade_names: Blade names to use as index

    Returns:
        blade_signs: signs for the passed blades in same order as passed
        blade_indices: blade indices in the same order as passed
    """
    signs_and_names = [get_normal_ordered(b) for b in blade_names]

    blade_signs = [sign for sign, blade_name in signs_and_names]

    blade_indices = [
        all_blade_names.index(blade_name) for sign, blade_name in signs_and_names
    ]

    return (torch.tensor(blade_signs, dtype=torch.float32),
            torch.tensor(blade_indices, dtype=torch.int64))

import math
from itertools import accumulate
def get_sub_algebra(n):
    """
    We have an algebra G(n) and we want to get G(n-1)
    it suppose to take the first element
    """
    idx_full = [math.comb(n, _) for _ in range(n+1)]
    idx_reduced = [math.comb(n-1, _) for _ in range(n)]
    pos_full = list(accumulate(idx_full))
    # pos_reduced = list(accumulate(idx_reduced))    
    pos_complex = [ pos+_  for pos,num in zip(pos_full,idx_reduced) for _ in  range(num)]
    pos_real = [ _  for _ in range(1<<n) if not _ in pos_complex]    
    return pos_real, pos_complex    
def get_sub_algebra_tree(n,k=1):
    """
    Returns the sub-algebra tree, 
    up to level 1<=k<=mn
    """
    assert(k<=n),f"should be k={k}<=n={n}"
    def _split(_list,n):
        def _filter(_list,_idx):
            return [_list[_] for _ in _idx]
        assert(len(_list)==1<<n),"should be correct size"
        if n<=1 or n<=k:
            return _list
        _r,_c = get_sub_algebra(n)
        _sr,_sc = _split(_filter(_list,_r),n-1),_split(_filter(_list,_c),n-1)
        # return [_list,_sr,_sc]
        return [_sr,_sc]
    _list = list(range(1<<n))
    return _split(_list,n)
def get_merged_tree(_tree,pairs_flag=True):
    """
    Flattens the tree
    """
    def _merge(_tmp):        
        if (pairs_flag and type(_tmp[0][0])==list) or (not pairs_flag and type(_tmp[0])==list):
            return _merge(_tmp[0])+_merge(_tmp[1])
        return _tmp
    _tree = _merge(_tree)
    return _tree

def get_complex_indexes(n):
    _tree = get_sub_algebra_tree(n)
    _list = get_merged_tree(_tree)
    return _list, _tree


def test():
    # tr = [0,2,3,4,8,9,10,14]
    # pr = [1,5,6,7,11,12,13,15]
    # print(get_sub_algebra(4))
    # print(get_sub_algebra(3))
    # print(get_sub_algebra(2))

    # print(get_sub_algebra(4))
    # print(get_sub_algebra_tree(4,3))
    
    _real,_complex = get_sub_algebra(4)
    print(_real,_complex)
    
    # _tree = get_sub_algebra_tree(4,2)
    _tree = get_sub_algebra_tree(4)
    
    # print(_tree)
    # print(len(_tree))

    print(get_merged_tree(_tree))
    # print(get_merged_tree(_tree,False))

    # ([0, 2, 3, 4, 8, 9, 10, 14], [1, 5, 6, 7, 11, 12, 13, 15])
    # [[0, 2, 3, 4, 8, 9, 10, 14], [1, 5, 6, 7, 11, 12, 13, 15]]
    # [[[[0, 4], [3, 10]], [[2, 9], [8, 14]]], [[[1, 7], [6, 13]], [[5, 12], [11, 15]]]]
    # 2
    # [0, 4, 3, 10, 2, 9, 8, 14, 1, 7, 6, 13, 5, 12, 11, 15]
    
# test()