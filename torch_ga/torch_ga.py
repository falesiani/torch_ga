"""Provides classes and operations for performing geometric algebra
with Pytorch.

The `GeometricAlgebra` class is used to construct the algebra given a metric.
It exposes methods for operating on `torch.Tensor` instances where their last
axis is interpreted as blades of the algebra.
"""
from typing import List, Any, Union, Optional
import numbers
import numpy as np
import torch
# import einops
from icecream import ic
import functools

from .cayley import get_cayley_tensor, blades_from_bases
from .blades import (
    BladeKind, get_blade_of_kind_indices, get_blade_indices_from_names,
    get_blade_repr, invert_blade_indices
)
from .mv_ops import mv_multiply, mv_reversion, mv_grade_automorphism, mv_conv1d, f_mv_conv1d, mv_multiply_element_wise
from .mv import MultiVector
from .blades import get_sub_algebra, get_sub_algebra_tree, get_merged_tree, get_complex_indexes

# from .utils import pga_meet,pga_join
# from opt_einsum.contract import contract

def int_comb(n,k):
    ntok = 1
    ktok = 1
    for t in range(1, min(k, n - k) + 1):
        ntok *= n
        ktok *= t
        n -= 1
    return ntok // ktok


# topar = lambda x: torch.nn.Parameter(torch.tensor(x) if isinstance(x,(list,tuple)) else x, requires_grad=False)
totorch = lambda x: torch.tensor(x) if isinstance(x,(list,tuple)) else x
topar = lambda x: torch.nn.Parameter(x, requires_grad=False)
ltopar = lambda x: [ topar(_) for _ in x]
class GeometricAlgebra:
    """Class used for performing geometric algebra operations on `torch.Tensor` instances.
    Exposes methods for operating on `torch.Tensor` instances where their last
    axis is interpreted as blades of the algebra.
    Holds the metric and other quantities derived from it.
    """

    def __init__(self, metric: List[float], compute_complex_flag=False, dtype=torch.float32, device="cpu"):
        super().__init__()
        """Creates a GeometricAlgebra object given a metric.
        The algebra will have as many basis vectors as there are
        elements in the metric.

        Args:
            metric: Metric as a list. Specifies what basis vectors square to
        """
        self.dtype = dtype
        self.device = device
        # self._metric = torch.tensor(metric, dtype=torch.float32)
        if isinstance(metric, torch.Tensor):
            self._metric = metric.detach()
        else:
            self._metric = torch.tensor(metric)
        

        self._num_bases = len(metric)
        self._dim = len(metric)
        self._num_bases_all = 1<<self._dim
        self._p,self._q,self._r = sum([1 for _ in metric if _>0]),sum([1 for _ in metric if _<0]),sum([1 for _ in metric if _==0])
        # self._bases = list(map(str, range(self._num_bases)))
        # Suggestion from TravisNP
        # self._bases = [chr(ord('a') + i) for i in range(self._num_bases)] 
        # let's extend a bit
        assert(self._num_bases<=26*2+10), f"We only support for number of bases <= {26*2+10}, but currently the number of bases={self._num_bases}."
        _i2c1 = lambda i : chr( (ord('a') + i if i<26 else ord('A') + i - 26) )
        i2c = lambda i : str(i) if i<10 else _i2c1(i-10)
        self._bases = list(map(i2c, range(self._num_bases)))

        self._blades, self._blade_degrees = blades_from_bases(self._bases)
        self._blade_degrees = torch.tensor(self._blade_degrees)
        
        self._even_grades = self._blade_degrees%2 == 0
        self._odd_grades =~self._even_grades
        self._num_blades = len(self._blades)
        self._max_degree = self._blade_degrees.max()

        # [Blades, Blades, Blades]
        _list = get_cayley_tensor(self.metric, self._bases, self._blades)
        # ic(_list)
        if type(_list) in [list,tuple]:
            _list = np.array(_list)
        self._cayley, self._cayley_inner, self._cayley_outer = torch.tensor(
            _list,
            dtype= self.dtype
            # dtype=torch.float32
        )

        self._blade_mvs = torch.eye(self._num_blades)
        self._basis_mvs = self._blade_mvs[1:1+self._num_bases]

        # Find the dual by looking at the anti-diagonal in the Cayley tensor.
        self._dual_blade_indices = []
        self._dual_blade_signs = []

        for blade_index in range(self._num_blades):
            dual_index = self.num_blades - blade_index - 1
            anti_diag = self._cayley[blade_index, dual_index]
            # dual_sign = tf.gather(anti_diag, tf.where(
            #     anti_diag != 0.0)[..., 0])[..., 0]
            dual_sign = anti_diag[torch.where(anti_diag != 0.0)]

            self._dual_blade_indices.append(dual_index)
            self._dual_blade_signs.append(dual_sign)

        self._dual_blade_indices = torch.tensor(
            self._dual_blade_indices, dtype=torch.int64)
        self._dual_blade_signs = torch.tensor(
            self._dual_blade_signs)
            # , dtype=torch.float32)
        
        self._I = self.e(self._blades[-1])
        self._Imv = self.emv(self._blades[-1])
        
        # complex indwxes
        self._real_idx,self._complex_idx = get_sub_algebra(self._dim)
        
        # all pairs of complex pairs, compute only if needed
        self._compute_complex_flag = compute_complex_flag
        self._complex_list, self._complex_tree = None, None
        if self._compute_complex_flag:
            self._complex_list, self._complex_tree = get_complex_indexes(self._dim)
            
       
        self.to(device)

    def to(self,device):
        self.device = device                
        self._blade_degrees = self._blade_degrees.to(device)
        self._even_grades = self._even_grades.to(device)
        self._odd_grades = self._odd_grades.to(device)
        self._max_degree = self._max_degree.to(device)
        self._cayley, self._cayley_inner, self._cayley_outer = self._cayley.to(device), self._cayley_inner.to(device), self._cayley_outer.to(device) 
        self._blade_mvs = self._blade_mvs.to(device)
        self._basis_mvs = self._basis_mvs.to(device)
        self._dual_blade_indices = self._dual_blade_indices.to(device)
        self._dual_blade_signs = self._dual_blade_signs.to(device)
        self._I = self._I.to(device)
        self._Imv = self._Imv.to(device)
        return self
        
    # def to_cl(self) -> CliffordAlgebra:
    #     """Creates a `CliffordAlgebra` from the Clifford Algebra

    #     Returns:
    #         `CliffordAlgebra` 
    #     """
    #     return CliffordAlgebra(self.metric)

    def to_cl(self):
        from torch_ga.clifford import CliffordAlgebra
        """Creates a `CliffordAlgebra` from the Clifford Algebra

        Returns:
            `CliffordAlgebra` 
        """
        return CliffordAlgebra(self.metric)

    def print(self, *args, **kwargs):
        """Same as the default `print` function but formats `torch.Tensor`
        instances that have as many elements on their last axis
        as the algebra has blades using `mv_repr()`.
        """
        def _is_mv(arg):
            return isinstance(arg, torch.Tensor) and len(arg.shape) > 0 and arg.shape[-1] == self.num_blades
        new_args = [self.mv_repr(arg) if _is_mv(arg) else arg for arg in args]

        print(*new_args, **kwargs)

    def split(self, a:torch.Tensor) -> list[torch.Tensor]:
        # return a.tensor.split( [int_comb(3,_) for _ in range(self.dim+1)],-1)
        return a.split( self.blades_numbers,-1)

    @property
    def blades_numbers(self) -> list[int]:
        """ G(n=(p+q+r), return n
        """
        return [int_comb(self.dim,_) for _ in range(self.dim+1)]


    @property
    def dim(self) -> int:
        """ G(n=(p+q+r), return n
        """
        return self._dim

    @property
    def p(self) -> int:
        """ G(p,q,r)
        """
        return self._p
    @property
    def q(self) -> int:
        """ G(p,q,r)
        """
        return self._q
    @property
    def r(self) -> int:
        """ G(p,q,r)
        """
        return self._r
    @property
    def dim(self) -> int:
        """ G(p,q,r|n=p+q+r)
        """
        return self._dim
    @property
    def num_bases(self) -> int:
        """ number of basis degree=1
        """
        return self._num_bases
    @property
    def num_blades(self) -> int:
        """ number of basis degree=1
        """
        return self._num_blades
    @property
    def num_coordinates(self) -> int:
        """ number of coordinates degree=1
        """
        return self._num_bases_all
        
    @property
    def real_idx(self)->list: return self._real_idx
    @property
    def complex_idx(self)->list: return self._complex_idx
    @property
    def complex_list(self)->list: return self._complex_list
    @property
    def complex_tree(self)->list: return self._complex_tree

    @property
    def metric(self) -> torch.Tensor:
        """Metric list which contains the number that each
        basis vector in the algebra squares to
        (ie. the diagonal of the metric tensor).
        """
        return self._metric

    @property
    def cayley(self) -> torch.Tensor:
        """`MxMxM` tensor where `M` is the number of basis
        blades in the algebra. Used for calculating the
        geometric product:

        `a_i, b_j, cayley_ijk -> c_k`
        """
        return self._cayley

    @property
    def cayley_inner(self) -> torch.Tensor:
        """Analagous to cayley but for inner product."""
        return self._cayley_inner

    @property
    def cayley_outer(self) -> torch.Tensor:
        """Analagous to cayley but for outer product."""
        return self._cayley_outer

    @property
    def blades(self) -> List[str]:
        """List of all blade names.

        Blades are all possible independent combinations of
        basis vectors. Basis vectors are named starting
        from `"0"` and counting up. The scalar blade is the
        empty string `""`.

        Example
        - Bases: `["0", "1", "2"]`
        - Blades: `["", "0", "1", "2", "01", "02", "12", "012"]`
        """
        return self._blades

    @property
    def blade_mvs(self) -> torch.Tensor:
        """List of all blade tensors in the algebra."""
        return self._blade_mvs

    @property
    def I(self) -> torch.Tensor:
        """List of all blade tensors in the algebra."""
        return self._I

    @property
    def Imv(self) -> MultiVector:
        """List of all blade tensors in the algebra."""
        return self._Imv

    @property
    def dual_blade_indices(self) -> torch.Tensor:
        """Indices of the dual blades for each blade."""
        return self._dual_blade_indices

    @property
    def dual_blade_signs(self) -> torch.Tensor:
        """Signs of the dual blades for each blade."""
        return self._dual_blade_signs

    @property
    def num_blades(self) -> int:
        """Total number of blades in the algebra."""
        return self._num_blades

    @property
    def blade_degrees(self) -> torch.Tensor:
        """List of blade-degree for each blade in the algebra."""
        return self._blade_degrees
    @property
    def even_grades(self) -> torch.Tensor:
        """List of even-degree blade in the algebra."""
        return self._even_grades
    @property
    def odd_grades(self) -> torch.Tensor:
        """List of even-degree blade in the algebra."""
        return self._odd_grades
    
    @property
    def max_degree(self) -> int:
        """Highest blade degree in the algebra."""
        return self._max_degree

    @property
    def basis_mvs(self) -> torch.Tensor:
        """List of basis vectors as torch.Tensor."""
        return self._basis_mvs

    def embed(self, tensor: torch.Tensor, tensor_index: torch.Tensor) -> torch.Tensor:
        mv = torch.zeros(
            *tensor.shape[:-1], 2**self.dim, device=tensor.device, dtype=tensor.dtype
        )
        mv[..., tensor_index] = tensor
        mv = mv.to(dtype=torch.float32)
        return mv

    def embed_grade(self, tensor: torch.Tensor, grade: int) -> torch.Tensor:
        mv = torch.zeros(*tensor.shape[:-1], 2**self.dim, device=tensor.device)
        s = self.grade_to_slice[grade]
        mv[..., s] = tensor
        return mv

    def get(self, mv: torch.Tensor, blade_index: tuple[int]) -> torch.Tensor:
        blade_index = tuple(blade_index)
        return mv[..., blade_index]

    def get_grade(self, mv: torch.Tensor, grade: int) -> torch.Tensor:
        s = self.grade_to_slice[grade]
        return mv[..., s]

    def get_kind_blade_indices(self, kind: BladeKind, invert: bool = False) -> torch.Tensor:
        """Find all indices of blades of a given kind in the algebra.

        Args:
            kind: kind of blade to give indices for
            invert: whether to return all blades not of the kind

        Returns:
            indices of blades of a given kind in the algebra
        """
        return get_blade_of_kind_indices(self.blade_degrees, kind, self.max_degree, invert=invert)

    def get_blade_indices_of_degree(self, degree: int) -> torch.Tensor:
        """Find all indices of blades of the given degree.

        Args:
            degree: degree to return blades for

        Returns:
            indices of blades with the given degree in the algebra
        """
        # return tf.gather(tf.range(self.num_blades), tf.where(self.blade_degrees == degree)[..., 0])
        return torch.range(self.num_blades)[torch.where(self.blade_degrees == degree)[..., 0]]

    def is_pure(self, tensor: torch.Tensor, blade_indices: torch.Tensor) -> bool:
        """Returns whether the given tensor is purely of the given blades
        and has no non-zero values for blades not in the given blades.

        Args:
            tensor: tensor to check purity for
            blade_indices: blade indices to check purity for

        Returns:
            Whether the tensor is purely of the given blades
            and has no non-zero values for blades not in the given blades
        """
        # tensor = torch.tensor(tensor, dtype=torch.float32)
        if False: tensor = tensor.to(dtype=torch.float32)
        if not type(blade_indices) in [torch.Tensor]:
            blade_indices = torch.tensor(blade_indices)
            
        blade_indices = blade_indices.to(dtype=torch.int64)

        # blade_indices = torch.tensor(
        #     blade_indices, dtype=torch.int64)

        inverted_blade_indices = invert_blade_indices(
            self.num_blades, blade_indices)

        # return tf.reduce_all(tf.gather(
        #     tensor,
        #     inverted_blade_indices,
        #     axis=-1
        # ) == 0)
        return (tensor[inverted_blade_indices]==0).sum(dim=-1)

    def is_pure_kind(self, tensor: torch.Tensor, kind: BladeKind) -> bool:
        """Returns whether the given tensor is purely of a given kind
        and has no non-zero values for blades not of the kind.

        Args:
            tensor: tensor to check purity for
            kind: kind of blade to check purity for

        Returns:
            Whether the tensor is purely of a given kind
            and has no non-zero values for blades not of the kind
        """
        # tensor = torch.tensor(tensor, dtype=torch.float32)
        if False: tensor = tensor.to(dtype=torch.float32)
        inverted_kind_indices = self.get_kind_blade_indices(kind, invert=True)
        # ic(inverted_kind_indices)
        # print(f"tensor={tensor}")
        # print(f"kind={kind}")
        # print(f"inverted_kind_indices={inverted_kind_indices.T}")
        # print(f"inverted_kind_indices.shape={inverted_kind_indices.shape}")
        # print(f"tensor[inverted_kind_indices]={tensor[inverted_kind_indices].T}")
        # print(f"tensor[inverted_kind_indices].shape={tensor[inverted_kind_indices].shape}")
        # print(f"tensor[inverted_kind_indices]==0={tensor[inverted_kind_indices].T==0}")

        # return tf.reduce_all(tf.gather(
        #     tensor,
        #     inverted_kind_indices,
        #     axis=-1
        # ) == 0)
        return (tensor[...,inverted_kind_indices]==0).sum(dim=-1)

    # def from_tensor(self, tensor: torch.Tensor, blade_indices: torch.Tensor) -> torch.Tensor:
    #     """Creates a geometric algebra torch.Tensor from a torch.Tensor and blade
    #     indices. The blade indices have to align with the last axis of the
    #     tensor.

    #     Args:
    #         tensor: torch.Tensor to take as values for the geometric algebra tensor
    #         blade_indices: Blade indices corresponding to the tensor. Can
    #         be obtained from blade names eg. using get_kind_blade_indices()
    #         or as indices from the blades list property.

    #     Returns:
    #         Geometric algebra torch.Tensor from tensor and blade indices
    #     """
    #     blade_indices = torch.tensor(blade_indices, dtype=torch.int64).to(dtype=torch.int64)
    #     tensor = torch.tensor(tensor, dtype=torch.float32)
    #     # print(f"blade_indices={blade_indices}")
    #     # print(f"tensor={tensor}")
        
    #     _shape = tensor.shape
    #     is_scalar = False
    #     if len(_shape)==1 :
    #         _shape_final = [1]+ [self.num_blades] 
    #         is_scalar = True
    #     else:
    #         _shape_final = list(_shape[:-1]) + [self.num_blades] 
    #     b = torch.zeros(_shape_final)
        

    #     # i = blade_indices.view([-1,1])
    #     # v = tensor.flatten().view([-1,1])
    #     i = blade_indices.nonzero().flatten()
    #     v = tensor.flatten().unsqueeze(1)
    #     b = b.view([-1,self.num_blades])
    #     # b[:,i] = v
    #     try:
    #         b[:,i] = v
    #     except:
    #         print(f"_shape={_shape},_shape_final={_shape_final}")
    #         print(f"i.shape={i.shape},v.shape={v.shape},b.shape={b.shape}")
    #         print(f"i={i},v={v},b={b}")
    #         raise
    #     #     raise "whatever"
    #     b = b.reshape(_shape_final)

    #     # _shape_tmp = list(v.shape) + [self.num_blades] 
    #     # print(f"i,v,_shape_tmp,_shape_final={i},{v},{_shape_tmp},{_shape_final},i.shape={i.shape}")
    #     # b = torch.sparse_coo_tensor(i, v, size=_shape_tmp)
    #     # print(f"b={b}")
    #     # b = torch.sparse_coo_tensor(i, v, size=_shape_tmp).to_dense()
    #     # b = b.reshape(_shape_final)
    #     if is_scalar:
    #         b=b.unsqueeze(0)
    #     return b

    #     # # Put last axis on first axis so scatter_nd becomes easier.
    #     # # Later undo the transposition again.
    #     # # t = tf.concat([[tensor.shape.ndims - 1],
    #     # #                tf.range(0, tensor.shape.ndims - 1)], axis=0)
    #     # # t_inv = tf.concat([tf.range(1, tensor.shape.ndims), [0]], axis=0)

    #     # # tensor = tf.transpose(tensor, t)

    #     # # shape = tf.concat([
    #     # #     torch.tensor([self.num_blades], dtype=torch.int64),
    #     # #     tf.shape(tensor, torch.int64)[1:]
    #     # # ], axis=0)

    #     # # tensor = tf.scatter_nd(
    #     # #     tf.expand_dims(blade_indices, axis=-1),
    #     # #     tensor,
    #     # #     shape
    #     # # )

    #     # # return tf.transpose(tensor, t_inv)
    #     # # t = torch.concat([torch.tensor([len(tensor.shape) - 1]), torch.range(0, len(tensor.shape)- 1)], axis=0)
    #     # # t_inv = torch.concat([torch.range(1, len(tensor.shape)), torch.tensor([0])], axis=0)
    #     # t = [len(tensor.shape) - 1] + list(range(0, len(tensor.shape)- 1))
    #     # t_inv = list(range(1, len(tensor.shape))) +  [0]

    #     # tensor = torch.permute(tensor, t)

    #     # a= torch.tensor([self.num_blades], dtype=torch.int64)
    #     # b = torch.tensor(tensor, dtype=torch.int64)[1:]
    #     # print("a,b:", a,b, tensor)


    #     # shape = torch.concat([
    #     #     torch.tensor([self.num_blades], dtype=torch.int64),
    #     #     torch.tensor(tensor, dtype=torch.int64)[1:]
    #     # ], axis=0)


    #     # # tensor = torch.scatter_nd(
    #     # #     blade_indices.unsqueeze(-1),
    #     # #     tensor,
    #     # #     shape
    #     # # )
    #     # a = torch.zeros(shape)
    #     # a[blade_indices] = tensor
    #     # tensor = a

    #     # return torch.permute(tensor, t_inv) 
            

    def from_tensor(self, tensor: torch.Tensor, blade_indices: torch.Tensor) -> torch.Tensor:
        """Creates a geometric algebra torch.Tensor from a torch.Tensor and blade
        indices. The blade indices have to align with the last axis of the
        tensor.

        Args:
            tensor: torch.Tensor to take as values for the geometric algebra tensor
            blade_indices: Blade indices corresponding to the tensor. Can
            be obtained from blade names eg. using get_kind_blade_indices()
            or as indices from the blades list property.

        Returns:
            Geometric algebra torch.Tensor from tensor and blade indices
        """
        # blade_indices = torch.tensor(blade_indices, dtype=torch.int64).to(dtype=torch.int64)
        # tensor = torch.tensor(tensor, dtype=torch.float32)
        blade_indices = blade_indices.to(dtype=torch.int64)
        if False: tensor = tensor.to(dtype=torch.float32)
        # print(f"blade_indices={blade_indices}")
        # print(f"tensor={tensor}")
        
        _shape = tensor.shape
        is_scalar = False
        if len(_shape)==1 :
            _shape_final = [1]+ [self.num_blades] 
            is_scalar = True
        else:
            _shape_final = list(_shape[:-1]) + [self.num_blades] 
        b = torch.zeros(_shape_final)

        if False:
            print(f"blade_indices.shape={blade_indices.shape}")
            print(f"tensor.shape={tensor.shape}")
            print(f"_shape_final={_shape_final}")
                


        # i = blade_indices.view([-1,1])
        # v = tensor.flatten().view([-1,1])
        # i = blade_indices.nonzero().flatten()
        i = blade_indices.flatten()
        # v = tensor.flatten().unsqueeze(1)
        v = tensor.view([-1,_shape[-1]])
        b = b.view([-1,self.num_blades])
        if False:
            print(f"_shape={_shape},_shape_final={_shape_final}")
            print(f"i.shape={i.shape},v.shape={v.shape},b.shape={b.shape}")
            print(f"i={i},v={v},b={b}")

        # b[:,i] = v
        try:
            b[:,i] = v
        except:
            print("Error:")
            print(f"_shape={_shape},_shape_final={_shape_final}")
            print(f"i.shape={i.shape},v.shape={v.shape},b.shape={b.shape}")
            # print(f"i={i},v={v},b={b}")
            raise
        b = b.reshape(_shape_final)

        if False:
            print(f"b.shape={b.shape}")

        if is_scalar:
            # b=b.unsqueeze(0)
            b=b.squeeze(0)
        return b


        # # i = blade_indices.view([-1,1])
        # # v = tensor.flatten().view([-1,1])
        # i = blade_indices.nonzero().flatten()
        # v = tensor.flatten().unsqueeze(1)
        # b = b.view([-1,self.num_blades])
        # # b[:,i] = v
        # try:
        #     b[:,i] = v
        # except:
        #     print(f"_shape={_shape},_shape_final={_shape_final}")
        #     print(f"i.shape={i.shape},v.shape={v.shape},b.shape={b.shape}")
        #     print(f"i={i},v={v},b={b}")
        #     raise
        # b = b.reshape(_shape_final)

        # if is_scalar:
        #     b=b.unsqueeze(0)
        # return b

       

    def from_tensor_with_kind(self, tensor: torch.Tensor, kind: BladeKind) -> torch.Tensor:
        """Creates a geometric algebra torch.Tensor from a torch.Tensor and a kind.
        The kind's blade indices have to align with the last axis of the
        tensor.

        Args:
            tensor: torch.Tensor to take as values for the geometric algebra tensor
            kind: Kind corresponding to the tensor

        Returns:
            Geometric algebra torch.Tensor from tensor and kind
        """
        # Put last axis on first axis so scatter_nd becomes easier.
        # Later undo the transposition again.
        # tensor = torch.tensor(tensor, dtype=torch.float32)
        if False: tensor = tensor.to(dtype=torch.float32)
        kind_indices = self.get_kind_blade_indices(kind)
        if False:
            print(f"tensor={tensor}")
            print(f"kind_indices={kind_indices}")
        return self.from_tensor(tensor, kind_indices)

    def from_scalar(self, scalar: numbers.Number) -> torch.Tensor:
        """Creates a geometric algebra torch.Tensor with scalar elements.

        Args:
            scalar: Elements to be used as scalars

        Returns:
            Geometric algebra torch.Tensor from scalars
        """
        # return self.from_tensor_with_kind(tf.expand_dims(scalar, axis=-1), BladeKind.SCALAR)
        # print("torch.tensor([scalar]).unsqueeze(-1).shape",torch.tensor([scalar]).unsqueeze(-1).shape)
        return self.from_tensor_with_kind(torch.tensor([scalar]).unsqueeze(-1), BladeKind.SCALAR).squeeze(0)

    def e(self, *blades: List[str]) -> torch.Tensor:
        """Returns a geometric algebra torch.Tensor with the given blades set
        to 1.

        Args:
            blades: list of blade names, can be unnormalized

        Returns:
            torch.Tensor with blades set to 1
        """
        blade_signs, blade_indices = get_blade_indices_from_names(
            blades, self.blades)
        blade_signs, blade_indices = blade_signs.to(self.device), blade_indices.to(self.device) 

        assert type(blade_indices) in [torch.Tensor], "should be a tensor"
        if False: blade_indices = torch.tensor(blade_indices)

        # # Don't allow duplicate indices
        # tf.Assert(
        #     blade_indices.shape[0] == tf.unique(blade_indices)[0].shape[0],
        #     [blades]
        # )

        # x = (
        #     tf.expand_dims(blade_signs, axis=-1) *
        #     tf.gather(self.blade_mvs, blade_indices)
        # )

        # # a, b -> b
        # return tf.reduce_sum(x, axis=-2)

        # print(f"blade_indices={blade_indices}")
        # print(f"torch.unique(blade_indices)={torch.unique(blade_indices)}")
        # print(f"torch.unique(blade_indices)[0]={torch.unique(blade_indices)[0]}")
        # Don't allow duplicate indices
        # assert(
        #     blade_indices.shape[0] == torch.unique(blade_indices).shape[0],
        #     [blades]
        # )
        assert blade_indices.shape[0] == torch.unique(blade_indices).shape[0], "indexes not unique"

        # ic = print
        # ic(blade_signs.device,self.blade_mvs.device)
        x = blade_signs.unsqueeze(-1) *  self.blade_mvs[blade_indices]

        # a, b -> b
        # return MultiVector(x.sum(dim=-2),self)
        return x.sum(dim=-2)
    def emv(self, *blades: List[str]) -> torch.Tensor:
        return MultiVector(self.e(*blades),self)      
    def bases(self,primal=True, names=False):
        ga = self
        if primal:
            _ret_bases= [ga.emv(_) for _ in ga._blades]
            if names:
                _ret_names = [f"e_{_}" for _ in ga._blades]
                return  _ret_bases, _ret_names
            else:
                return  _ret_bases        
        else:
            _ret_bases= [ga.emv(_).dual() for _ in ga._blades]
            if names:
                _ret_names = [f"e^{_}" for _ in ga._blades]
                return  _ret_bases, _ret_names
            else:
                return  _ret_bases        
    def bases_primal(self,names=False):
        return self.bases(primal=True,names=names)
    def bases_dual(self,names=False):
        return self.bases(primal=False,names=names)

    # def grade(self,tensor):
    #     """
    #     Returns the max grade of all non zero blades.
    #     """
    #     ga = self
    #     _grade = torch.einsum("...i,i->...i",tensor,ga.blade_degrees).max(-1)
    #     return _grade
    
    def grade(self,tensor,_grade=None):
        ga = self
        if _grade is None: # return the degree of the max blade
            _grade = torch.einsum("...i,i->...i",tensor,ga.blade_degrees).max(-1)[0]
            return _grade
        if _grade < 0: _grade+=max(ga.blade_degrees+1)
        if isinstance(tensor,torch.Tensor):
            return tensor*(ga.blade_degrees==_grade).float()
        if isinstance(tensor,MultiVector):
            return tensor.tensor*(ga.blade_degrees==_grade).float()
        return None
    
    def __getattr__(self, name: str) -> torch.Tensor:
        """Returns basis blade tensors if name was a basis."""
        if name.startswith("e") and (name[1:] == "_" or name[1:] == "" or int(name[1:]) >= 0):
            return self.e( [""] if name[1:]=="_" else name[1:] )
        raise AttributeError(f"{name}")

    def dual(self, tensor: torch.Tensor) -> torch.Tensor:
        """Returns the dual of the geometric algebra tensor.

        Args:
            tensor: Geometric algebra tensor to return dual for

        Returns:
            Dual of the geometric algebra tensor
        """
        if not isinstance(tensor,torch.Tensor):
            tensor = torch.tensor(tensor)
            # tensor = torch.tensor(tensor, dtype=torch.float32)

        # else:
        #     ic(f"detaching!!! {type(tensor)}")
        #     tensor = tensor.clone().detach().requires_grad_(True) if tensor.requires_grad else tensor.clone().detach()

        # return self.dual_blade_signs * tf.gather(tensor, self.dual_blade_indices, axis=-1)
        return self.dual_blade_signs * tensor[...,self.dual_blade_indices]

    def grade_automorphism(self, tensor: torch.Tensor) -> torch.Tensor:
        """Returns the geometric algebra tensor with odd grades negated.
        See https://en.wikipedia.org/wiki/Paravector#Grade_automorphism.

        Args:
            tensor: Geometric algebra tensor to return grade automorphism for

        Returns:
            Geometric algebra tensor with odd grades negated
        """
        if False: tensor = tensor.to(dtype=torch.float32)
        return mv_grade_automorphism(tensor, self.blade_degrees)

    def reversion(self, tensor: torch.Tensor) -> torch.Tensor:
        """Returns the grade-reversed geometric algebra tensor.
        See https://en.wikipedia.org/wiki/Paravector#Reversion_conjugation.

        Args:
            tensor: Geometric algebra tensor to return grade-reversion for

        Returns:
            Grade-reversed geometric algebra tensor
        """
        if False: tensor = tensor.to(dtype=torch.float32)

        return mv_reversion(tensor, self.blade_degrees)

    def conjugation(self, tensor: torch.Tensor) -> torch.Tensor:
        """Combines reversion and grade automorphism.
        See https://en.wikipedia.org/wiki/Paravector#Clifford_conjugation.

        Args:
            tensor: Geometric algebra tensor to return conjugate for

        Returns:
            Geometric algebra tensor after `reversion()` and `grade_automorphism()`
        """
        if False: tensor = tensor.to(dtype=torch.float32)
        return self.grade_automorphism(self.reversion(tensor))

    def simple_inverse(self, a: torch.Tensor) -> torch.Tensor:
        """Returns the inverted geometric algebra tensor
        `X^-1` such that `X * X^-1 = 1`. Only works for elements that
        square to scalars. Faster than the general inverse.

        Args:
            a: Geometric algebra tensor to return inverse for

        Returns:
            inverted geometric algebra tensor
        """
        if False: a = a.to(dtype=torch.float32)


        rev_a = self.reversion(a)
        divisor = self.geom_prod(a, rev_a)
        # print(f"divisor={divisor}")
        # print(f"self.is_pure_kind(divisor, BladeKind.SCALAR)={self.is_pure_kind(divisor, BladeKind.SCALAR)}")
        if not self.is_pure_kind(divisor, BladeKind.SCALAR):
            raise Exception(
                "Can't invert multi-vector (inversion divisor V ~V not scalar: %s)." % divisor)

        # Divide by scalar part
        return rev_a / divisor[..., :1]

    def reg_prod(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Returns the regressive product of two geometric
        algebra tensors.

        Args:
            a: Geometric algebra tensor on the left hand side of
            the regressive product
            b: Geometric algebra tensor on the right hand side of
            the regressive product

        Returns:
            regressive product of a and b
        """
        if not isinstance(a,torch.Tensor): a = torch.tensor(a, dtype=torch.float32)
        if not isinstance(b,torch.Tensor): b = torch.tensor(b, dtype=torch.float32)

        return self.dual(self.ext_prod(self.dual(a), self.dual(b)))
    
    def left_contraction(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Returns the left constraction of two geometric
        algebra tensors.

        Args:
            a: Geometric algebra tensor on the left hand side of
            the regressive product
            b: Geometric algebra tensor on the right hand side of
            the regressive product

        Returns:
            left contraction of a and b
        """
        if not isinstance(a,torch.Tensor): a = torch.tensor(a, dtype=torch.float32)
        if not isinstance(b,torch.Tensor): b = torch.tensor(b, dtype=torch.float32)

        return self.dual(self.ext_prod(a, self.dual(b)))
    
    def right_contraction(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Returns the right constraction of two geometric
        algebra tensors.

        Args:
            a: Geometric algebra tensor on the left hand side of
            the regressive product
            b: Geometric algebra tensor on the right hand side of
            the regressive product

        Returns:
            right contraction of a and b
        """
        if not isinstance(a,torch.Tensor): a = torch.tensor(a, dtype=torch.float32)
        if not isinstance(b,torch.Tensor): b = torch.tensor(b, dtype=torch.float32)

        return self.dual(self.ext_prod(self.dual(a), b))

    def projection(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Returns the projection of A on B, two geometric
        algebra tensors.

        Args:
            a: Geometric algebra tensor on the left hand side of
            the regressive product
            b: Geometric algebra tensor on the right hand side of
            the regressive product

        Returns:
            projection of a and b: P_B(A)
        """
        if not isinstance(a,(torch.Tensor,MultiVector)): a = torch.tensor(a, dtype=torch.float32)
        if not isinstance(b,(torch.Tensor,MultiVector)): b = torch.tensor(b, dtype=torch.float32)

        # return self.left_contraction(self.left_contraction(a,self.inverse(b)),b)
        return self.left_contraction(self.left_contraction(a,b),self.inverse(b))

    def orthogonal_projection(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Returns the orthogonal projection of A on B, two geometric
        algebra tensors.

        Args:
            a: Geometric algebra tensor on the left hand side of
            the regressive product
            b: Geometric algebra tensor on the right hand side of
            the regressive product

        Returns:
            orthogonal projection of a and b: P_B(A)
        """
        if not isinstance(a,torch.Tensor): a = torch.tensor(a, dtype=torch.float32)
        if not isinstance(b,torch.Tensor): b = torch.tensor(b, dtype=torch.float32)

        # return self.left_contraction(self.left_contraction(a,self.inverse(b)),b)
        return a-self.projection(a,b)


    def ext_prod(self, a: torch.Tensor, b: torch.Tensor, blades=None) -> torch.Tensor:
        """Returns the exterior product of two geometric
        algebra tensors.

        Args:
            a: Geometric algebra tensor on the left hand side of
            the exterior product
            b: Geometric algebra tensor on the right hand side of
            the exterior product

        Returns:
            exterior product of a and b
        """
        if False: a = a.to(dtype=torch.float32)
        if False: b = b.to(dtype=torch.float32)        
        # cayley = self._cayley_outer
        # if blades is not None:
        #     blades_l, blades_o, blades_r = blades
        #     assert isinstance(blades_l, torch.Tensor)
        #     assert isinstance(blades_o, torch.Tensor)
        #     assert isinstance(blades_r, torch.Tensor)
        #     cayley = cayley[blades_l[:, None, None], blades_o[:, None], blades_r]
        # return torch.einsum("...i,ijk,...k->...j", a, cayley, b)
        return mv_multiply(a, b, self._cayley_outer)

    def geom_prod(self, a: torch.Tensor, b: torch.Tensor, blades=None) -> torch.Tensor:
        """Returns the geometric product of two geometric
        algebra tensors.

        Args:
            a: Geometric algebra tensor on the left hand side of
            the geometric product
            b: Geometric algebra tensor on the right hand side of
            the geometric product

        Returns:
            geometric product of a and b
        """
        # a = torch.tensor(a, dtype=torch.float32)
        # b = torch.tensor(b, dtype=torch.float32)

        # a = torch.tensor(a)
        # b = torch.tensor(b)

        # if False: a = a.to(dtype=torch.float32)
        # if False: b = b.to(dtype=torch.float32)
        # return mv_multiply(a, b, self._cayley)
    
        # cayley = self._cayley
        # if blades is not None:
        #     blades_l, blades_o, blades_r = blades
        #     assert isinstance(blades_l, torch.Tensor)
        #     assert isinstance(blades_o, torch.Tensor)
        #     assert isinstance(blades_r, torch.Tensor)
        #     cayley = cayley[blades_l[:, None, None], blades_o[:, None], blades_r]
        # return torch.einsum("...i,ijk,...k->...j", a, cayley, b)    
        return mv_multiply(a, b, self._cayley)

    # # start
    # def q(self, mv, blades=None):
    #     if blades is not None:
    #         blades = (blades, blades)
    #     return self.b(mv, mv, blades=blades)
        
    # def alpha_w(self, w, mv):
    #     return self.even_grades * mv + self.eta(w) * self.odd_grades * mv

    # # def inverse(self, mv, blades=None):
    # #     mv_ = self.beta(mv, blades=blades)
    # #     return mv_ / self.q(mv)
    
    # def rho(self, w, mv):
    #     """Applies the versor w action to mv."""
    #     return self.sandwich(w, self.alpha_w(w, mv), self.inverse(w))
    # # end

    def pga_meet(self,A,B):
        return pga_meet(A,B)

    def pga_join(self,A,B):
        return pga_join(self,None,A,B)
    
    def meet(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # it is actually the wedge
        return self.ext_prod(a,b)

    def join(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # A Guided Tour to the Plane-Based Geometric Algebra PGA
        # assert(self.r==0),"this join does not work well, please use pga_join() function"
        if False: a = a.to(dtype=torch.float32)
        if False: b = b.to(dtype=torch.float32)    
        
        b = self.dual(b)
        a = self.dual(a)
        c = mv_multiply(b, a, self._cayley_outer)
        c = self.inv(c, force_scalar=True)
        c = self.dual(c)
        return c
    
    def sandwitch_prod(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Returns the geometric sandwitch product of two geometric
        algebra tensors.

        Args:
            a: Geometric algebra tensor on the left hand side of
            the geometric product
            b: Geometric algebra tensor on the right hand side of
            the geometric product

        Returns:
            geometric product of b a b~
        """
        # a = torch.tensor(a, dtype=torch.float32)
        # b = torch.tensor(b, dtype=torch.float32)

        # a = torch.tensor(a)
        # b = torch.tensor(b)

        if False: a = a.to(dtype=torch.float32)
        if False: b = b.to(dtype=torch.float32)
        
        ab = mv_multiply(a,self.reversion(b), self._cayley)
        bab = mv_multiply(b, ab, self._cayley)
        return bab

    
    def element_wise_prod(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Returns the element-wise product of two geometric
        algebra tensors.

        Args:
            a: Geometric algebra tensor on the left hand side of
            the geometric product
            b: Geometric algebra tensor on the right hand side of
            the geometric product

        Returns:
            geometric product of a and b
        """
        # a = torch.tensor(a, dtype=torch.float32)
        # b = torch.tensor(b, dtype=torch.float32)

        # a = torch.tensor(a)
        # b = torch.tensor(b)

        if False: a = a.to(dtype=torch.float32)
        if False: b = b.to(dtype=torch.float32)
        return mv_multiply_element_wise(a, b, self._cayley)


    def inner_prod(self, a: torch.Tensor, b: torch.Tensor, blades= None) -> torch.Tensor:
        """Returns the inner product of two geometric
        algebra tensors.

        Args:
            a: Geometric algebra tensor on the left hand side of
            the inner product
            b: Geometric algebra tensor on the right hand side of
            the inner product

        Returns:
            inner product of a and b
        """
        if False: a = a.to(dtype=torch.float32)
        if False: b = b.to(dtype=torch.float32)
        # cayley = self._cayley_inner
        # if blades is not None:
        #     blades_l, blades_o, blades_r = blades
        #     assert isinstance(blades_l, torch.Tensor)
        #     assert isinstance(blades_o, torch.Tensor)
        #     assert isinstance(blades_r, torch.Tensor)
        #     cayley = cayley[blades_l[:, None, None], blades_o[:, None], blades_r]
        # return torch.einsum("...i,ijk,...k->...j", a, cayley, b)
        return mv_multiply(a, b, self._cayley_inner)

    def geom_conv1d(self, a: torch.Tensor, k: torch.Tensor,
                    stride: int, padding: str,
                    dilations: Union[int, None] = None) -> torch.Tensor:
        """Returns the 1D convolution of a sequence with a geometric algebra
        tensor kernel. The convolution is performed using the geometric
        product.

        Args:
            a: Input geometric algebra tensor of shape
                [..., Length, ChannelsIn, Blades]
            k: Geometric algebra tensor for the convolution kernel of shape
                [KernelSize, ChannelsIn, ChannelsOut, Blades]
            stride: Stride to use for the convolution
            padding: "SAME" (zero-pad input length so output
                length == input length / stride) or "VALID" (no padding)
        Returns:
            Geometric algbra tensor of shape
            [..., OutputLength, ChannelsOut, Blades]
            representing `a` convolved with `k`
        """
        if False: a = a.to(dtype=torch.float32)
        if False: k = k.to(dtype=torch.float32)

        # return mv_conv1d(a, k, self._cayley, stride=stride, padding=padding)
        return f_mv_conv1d(a, k, self._cayley, stride=stride, padding=padding)

    def mv_repr(self, a: torch.Tensor, prefix="") -> str:
        """Returns a string representation for the given
        geometric algebra tensor.

        Args:
            a: Geometric algebra tensor to return the representation for

        Returns:
            string representation for `a`
        """
        if False: a = a.to(dtype=torch.float32)

        if len(a.shape) == 1:
            _scalar_name=''
            # _str = f'{prefix}{" + ".join([f"{value:g}{get_blade_repr(blade_name,_scalar_name)}" for value, blade_name in zip(a, self.blades) if value != 0])}'
            _str = f'{prefix}{" + ".join([f"{value:g}{get_blade_repr(blade_name,_scalar_name)}" for value, blade_name in zip(a, self.blades) if abs(value) > 1e-6])}'
            if _str==f'{prefix}':
                return f'{prefix}{0}'
            else:
                return _str
            
        else:
            return f"MV[batch_shape={a.shape[:-1]}]"


        # if len(a.shape) == 1:
        #     return "MultiVector[%s]" % " + ".join(
        #         "%.2f*%s" % (value, get_blade_repr(blade_name))
        #         for value, blade_name
        #         in zip(a, self.blades)
        #         if value != 0
        #     )
        # else:
        #     return f"MultiVector[batch_shape={a.shape[:-1]}]"

    def approx_exp(self, a: torch.Tensor, order: int = 50) -> torch.Tensor:
        """Returns an approximation of the exponential using a centered taylor series.

        Args:
            a: Geometric algebra tensor to return exponential for
            order: order of the approximation

        Returns:
            Approximation of `exp(a)`
        """
        if False: a = a.to(dtype=torch.float32)

        v = self.from_scalar(1.0)
        result = self.from_scalar(1.0)
        for i in range(1, order + 1):
            v = self.geom_prod(a, v)
            # i_factorial = tf.exp(tf.math.lgamma(i + 1.0))
            i_factorial = torch.exp(torch.lgamma(torch.tensor([i + 1.0])))
            result += v / i_factorial
        return result

    def exp(self, a: torch.Tensor, square_scalar_tolerance: Union[float, None] = 1e-4) -> torch.Tensor:
        """Returns the exponential of the passed geometric algebra tensor.
        Only works for multivectors that square to scalars.

        Args:
            a: Geometric algebra tensor to return exponential for
            square_scalar_tolerance: Tolerance to use for the square scalar check
                or None if the check should be skipped

        Returns:
            `exp(a)`
        """
        # See https://www.euclideanspace.com/maths/algebra/clifford/algebra/functions/exponent/index.htm
        # for an explanation of how to exponentiate multivectors.

        self_sq = self.geom_prod(a, a)

        if square_scalar_tolerance is not None:
            # tf.Assert(tf.reduce_all(
            #     tf.abs(self_sq[..., 1:]) < square_scalar_tolerance
            # ), [self_sq])
            
            # assert torch.equal(torch.all(self_sq[..., 1:].abs() < square_scalar_tolerance),[self_sq]), "not sure what"
            assert torch.all(self_sq[..., 1:].abs() < square_scalar_tolerance), "square_scalar_tolerance not met"

        scalar_self_sq = self_sq[..., :1]

        # "Complex" square root (argument can be negative)
        s_sqrt = torch.sign(scalar_self_sq) * torch.sqrt(torch.abs(scalar_self_sq))

        # Square to +1: cosh(sqrt(||a||)) + a / sqrt(||a||) sinh(sqrt(||a||))
        # Square to -1: cos(sqrt(||a||)) + a / sqrt(||a||) sin(sqrt(||a||))
        # TODO: Does this work for values other than 1 too? eg. square to +0.5?
        # TODO: Find a solution that doesnt require calculating all possibilities
        #       first.
        non_zero_result = torch.where(
            scalar_self_sq < 0,
            (self.from_tensor(torch.cos(s_sqrt), torch.tensor([0])) +  a / s_sqrt * torch.sin(s_sqrt)),
            (self.from_tensor(torch.cosh(s_sqrt), torch.tensor([0])) +  a / s_sqrt * torch.sinh(s_sqrt))
        )

        return torch.where(scalar_self_sq == 0, self.from_scalar(1.0) + a, non_zero_result)

    def approx_log(self, a: torch.Tensor, order: int = 50) -> torch.Tensor:
        """Returns an approximation of the natural logarithm using a centered
        taylor series. Only converges for multivectors where `||mv - 1|| < 1`.

        Args:
            a: Geometric algebra tensor to return logarithm for
            order: order of the approximation

        Returns:
            Approximation of `log(a)`
        """
        if False: a = a.to(dtype=torch.float32)

        result = self.from_scalar(0.0)

        a_minus_one = a - self.from_scalar(1.0)
        v = None

        for i in range(1, order + 1):
            v = a_minus_one if v is None else v * a_minus_one
            result += (((-1.0) ** i) / i) * v

        return -result

    def int_pow(self, a: torch.Tensor, n: int) -> torch.Tensor:
        """Returns the geometric algebra tensor to the power of an integer
        using repeated multiplication.

        Args:
            a: Geometric algebra tensor to raise
            n: integer power to raise the multivector to

        Returns:
            `a` to the power of `n`
        """
        if False: a = a.to(dtype=torch.float32)


        if not isinstance(n, int):
            raise Exception("n must be an integer.")
        if n < 0:
            raise Exception("Can't raise to negative powers.")

        if n == 0:
            # TODO: more efficient (ones only in scalar)
            return torch.ones_like(a) * self.e("")

        result = a
        for i in range(n - 1):
            result = self.geom_prod(result, a)
        return result

    def keep_blades(self, a: torch.Tensor, blade_indices: List[int]) -> torch.Tensor:
        """Takes a geometric algebra tensor and returns it with only the given
        blade_indices as non-zeros.

        Args:
            a: Geometric algebra tensor to copy
            blade_indices: Indices for blades to keep

        Returns:
            `a` with only `blade_indices` components as non-zeros
        """
        if False: a = a.to(dtype=torch.float32)
        blade_indices = blade_indices.to(dtype=torch.int64)

        # blade_values = tf.gather(a, blade_indices, axis=-1)
        blade_values = a[...,blade_indices]
        if True: 
            b = self.from_tensor(blade_values, blade_indices)
        else:
            blade_mask = torch.zeros(self.num_blades)
            blade_mask[blade_indices] = 1
            b = self.from_tensor(blade_values, blade_mask)
        # print(f"blade_values, blade_indices, b={blade_values}, {blade_indices}, {b}")
        # print(f"blade_mask={blade_mask}")
        return b

        # return self.from_tensor(blade_values, blade_indices)

    def keep_blades_with_name(self, a: torch.Tensor, blade_names: Union[List[str], str]) -> torch.Tensor:
        """Takes a geometric algebra tensor and returns it with only the given
        blades as non-zeros.

        Args:
            a: Geometric algebra tensor to copy
            blade_names: Blades to keep

        Returns:
            `a` with only `blade_names` components as non-zeros
        """
        if isinstance(blade_names, str):
            blade_names = [blade_names]

        _, blade_indices = get_blade_indices_from_names(blade_names, self.blades)

        if False:
            print(f"self.blades={self.blades}")
            print(f"blade_names={blade_names}")
            print(f"blade_indices={blade_indices}")

        return self.keep_blades(a, blade_indices)

    def select_blades(self, a: torch.Tensor, blade_indices: List[int]) -> torch.Tensor:
        """Takes a geometric algebra tensor and returns a `torch.Tensor` with the
        blades in blade_indices on the last axis.


        Args:
            a: Geometric algebra tensor to copy
            blade_indices: Indices for blades to select

        Returns:
            `torch.Tensor` based on `a` with `blade_indices` on last axis.
        """
        if False: a = a.to(dtype=torch.float32)        
        # blade_indices = torch.tensor(blade_indices, dtype=torch.int64).to(dtype=torch.int64)
        blade_indices = blade_indices.to(dtype=torch.int64)

        # result = tf.gather(a, blade_indices, axis=-1)
        try:
            if len(a.shape)==1 or a.shape[-1]==a.size().numel():
                result = a.squeeze()[blade_indices]
            else:
                result = a[...,blade_indices]
        except:
            print(f"a={a},blade_indices={blade_indices}")
            print(f"a.shape={a.shape},blade_indices.shape={blade_indices.shape},a.size().numel()={a.size().numel()}")
            raise
        
        return result

    def select_blades_with_name(self, a: torch.Tensor, blade_names: Union[List[str], str]) -> torch.Tensor:
        """Takes a geometric algebra tensor and returns a `torch.Tensor` with the
        blades in blade_names on the last axis.


        Args:
            a: Geometric algebra tensor to copy
            blade_names: Blades to keep

        Returns:
            `torch.Tensor` based on `a` with `blade_names` on last axis.
        """
        if False: a = a.to(dtype=torch.float32)

        is_single_blade = isinstance(blade_names, str)
        if is_single_blade:
            blade_names = [blade_names]

        blade_signs, blade_indices = get_blade_indices_from_names(
            blade_names, self.blades)

        result = blade_signs * self.select_blades(a, blade_indices)
        # if True:
        #     print(f"")

        if is_single_blade:
            return result[..., 0]

        return result

    def inverse(self, a: torch.Tensor, force_scalar=False) -> torch.Tensor:
        """Returns the inverted geometric algebra tensor
        `X^-1` such that `X * X^-1 = 1`.

        Using Shirokov's inverse algorithm that works in arbitrary dimensions,
        see https://arxiv.org/abs/2005.04015 Theorem 4.

        Args:
            a: Geometric algebra tensor to return inverse for

        Returns:
            inverted geometric algebra tensor
        """
        # a = torch.tensor(a, dtype=torch.float32)
        if False: a = a.to(dtype=torch.float32)
        if False:
            print(f"a={a}")

        n = 2 ** ((len(self.metric) + 1) // 2)

        # u = a.clone()
        u = a
        for k in range(1, n):
            # c = n / k * self.keep_blades_with_name(u, "")
            d = self.keep_blades_with_name(u, "")
            c = n / k * d
            u_minus_c = u - c
            if False:
                print(f"a,d,c,u_minus_c, u = {a},{d},{c},{u_minus_c}, {u}")
            u = self.geom_prod(a, u_minus_c)
            if False:
                print(f"u={u}")
        
        if False:
            print(f"n={n}")
            print(f"a={a}")
            print(f"u={u}")
        if force_scalar:
            u[...,1:] = 0.
        if not torch.all(self.is_pure_kind(u, BladeKind.SCALAR)):
            raise Exception(f"Can't invert multi-vector (det U not scalar: {u}).")
                # "Can't invert multi-vector (det U not scalar: %s)." % u.tensor)

        # adj / det
        return u_minus_c / u[..., :1]
    
    def inv(self, a: torch.Tensor, **kargs) -> torch.Tensor:
        return self.inverse(a, **kargs)
    
    def prod(self,a,b):
        return self.geom_prod(a,b)
    def ext(self,a,b):
        return self.ext_prod(a,b)
    def inner(self,a,b):
        return self.inner_prod(a,b)
    
    def norm(self,a):
        return abs(self.geom_prod(a, self.conjugation(a))[...,0])**0.5        
    def inorm(self,a):
        return self.norm(self.dual(a))        
    def normalized(self,a):
        if len(a.shape)==1:
            return a * (1 / self.norm(a))
        else:
            return a * (1 / self.norm(a).unsqueeze(-1))
    
    
    
        

    def __call__(self, a: torch.Tensor) -> MultiVector:
        """Creates a `MultiVector` from a geometric algebra tensor.
        Mainly used as a wrapper for the algebra's functions for convenience.

        Args:
            a: Geometric algebra tensor to return `MultiVector` for

        Returns:
            `MultiVector` for `a`
        """
        if False: a = a.to(dtype=torch.float32)
        return MultiVector(a, self)
        # return MultiVector(torch.tensor(a), self)




# # PGA specific function for PGA(3d)
# def pga2tp(ga, tensor):
#     tr = [0,2,3,4,8,9,10,14]
#     pr = [1,5,6,7,11,12,13,15]

#     Ta = tensor.clone()
#     Ta[...,pr] = 0.

#     Pa = tensor.clone()
#     Pa[...,tr] = Pa[...,pr]
#     Pa[...,pr] = 0.
#     return Ta,Pa

# def tp2pga(ga,Ta,Pa):
#     return Ta + ga.geom_prod(ga.e0, Pa)

# # Special operation for the PGA
# def pga_meet(ga,A,B):
#     return ga.meet(A,B)

# def pga_join(ga, A, B):
#     Ta,Pa = pga2tp(ga, A)
#     Tb,Pb = pga2tp(ga, B)    
#     d = ga.p
#     # (-1)**ga.grade(Pa)
#     # AvB = ga.join(Ta,Tb) + (-1)**d*ga.join(Pa,Tb) + ga.prod_geom(ga.e0,ga.join(Pa,Pb))
#     AvB = tp2pga(ga,ga.join(Ta,Tb) + (-1)**d*ga.join(Pa,Tb),ga.join(Pa,Pb))
#     return AvB




# PGA specific function for PGA(3d)
def pga2tp(ga, tensor):
    # tr = [0,2,3,4,8,9,10,14]
    # pr = [1,5,6,7,11,12,13,15]
    assert(tensor.shape[-1]==ga.num_blades), f"tensor dimension do not match {tensor.shape}, {ga.num_blades}"
    
    tr = ga.real_idx
    pr = ga.complex_idx
    
    
    Ta = tensor.clone()
    # Ta[...,pr] = 0.
    Ta = Ta[...,tr]

    Pa = tensor.clone()
    # Pa[...,tr] = Pa[...,pr]
    # Pa[...,pr] = 0.
    Pa = Pa[...,pr]
    return Ta,Pa

# def tp2pga(ga,Ta,Pa):
#     tr = [0,2,3,4,8,9,10,14]
#     pr = [1,5,6,7,11,12,13,15]
#     _Ta = torch.zeros([*Ta.shape[:-1],16])
#     _Ta[...,tr] = Ta
#     _Pa = torch.zeros([*Pa.shape[:-1],16])
#     _Pa[...,tr] = Pa    
#     return _Ta + ga.geom_prod(ga.e0, _Pa)

# def tp2pga_v2(ga,Ta,Pa):
#     tr = [0,2,3,4,8,9,10,14]
#     pr = [1,5,6,7,11,12,13,15]
#     _Ta = torch.zeros([*Ta.shape[:-1],16])
#     _Ta[...,tr] = Ta
#     _Pa = torch.zeros([*Pa.shape[:-1],16])
#     _Pa[...,tr] = Pa    
#     return _Ta + ga.geom_prod(ga.e0, _Pa)

def tp2pga(ga,Ta,Pa):
    # tr = [0,2,3,4,8,9,10,14]
    # pr = [1,5,6,7,11,12,13,15]
    tr = ga.real_idx
    pr = ga.complex_idx
    
    A = torch.zeros([*Ta.shape[:-1],16])
    A[...,tr] = Ta
    A[...,pr] = Pa    
    # assert(((A-tp2pga_v2(ga,Ta,Pa))**2).sum()==0.),f" should be the same {A},{tp2pga_v2(ga,Ta,Pa)}"
    return A

# Special operation for the PGA
def pga_meet(ga,A,B):
    return ga.meet(A,B)

def pga_join(ga, A, B, ga_d = None):
    Ta,Pa = pga2tp(ga, A)
    Tb,Pb = pga2tp(ga, B)
    # ic(Ta,Pa,Tb,Pb)
    d = ga.p + ga.q
    if ga_d is None:
        ga_d = GeometricAlgebra(ga.p*[1]+ga.q*[-1])
    
    # (-1)**ga.grade(Pa)
    # AvB = ga.join(Ta,Tb) + (-1)**d*ga.join(Pa,Tb) + ga.prod_geom(ga.e0,ga.join(Pa,Pb))
    
    # this should work
    # AvB = tp2pga(ga,ga_d.join(Ta,Tb) + (-1)**d*ga_d.join(Pa,Tb),ga_d.join(Pa,Pb))
    
    # this should work: alternative
    Ta,Pa,Tb,Pb = ga_d(Ta),ga_d(Pa),ga_d(Tb),ga_d(Pb)
    
    AvB = tp2pga(ga,(Ta.join(Tb)).tensor + (-1)**d*(Pa.join(Tb)).tensor,(Pa.join(Pb)).tensor)
    
    
    return AvB