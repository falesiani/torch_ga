"""Provides classes and operations for performing geometric algebra
with TensorFlow.

The `GeometricAlgebra` class is used to construct the algebra given a metric.
It exposes methods for operating on `torch.Tensor` instances where their last
axis is interpreted as blades of the algebra.
"""
from typing import List, Any, Union, Optional
import numbers
import numpy as np
import torch
# import einops

from .cayley import get_cayley_tensor, blades_from_bases
from .blades import (
    BladeKind, get_blade_of_kind_indices, get_blade_indices_from_names,
    get_blade_repr, invert_blade_indices
)
from .mv_ops import mv_multiply, mv_reversion, mv_grade_automorphism, mv_conv1d, f_mv_conv1d, mv_multiply_element_wise
from .mv import MultiVector


class GeometricAlgebra:
    """Class used for performing geometric algebra operations on `torch.Tensor` instances.
    Exposes methods for operating on `torch.Tensor` instances where their last
    axis is interpreted as blades of the algebra.
    Holds the metric and other quantities derived from it.
    """

    def __init__(self, metric: List[float]):
        """Creates a GeometricAlgebra object given a metric.
        The algebra will have as many basis vectors as there are
        elements in the metric.

        Args:
            metric: Metric as a list. Specifies what basis vectors square to
        """
        self._metric = torch.tensor(metric, dtype=torch.float32)

        self._num_bases = len(metric)
        self._bases = list(map(str, range(self._num_bases)))

        self._blades, self._blade_degrees = blades_from_bases(self._bases)
        self._blade_degrees = torch.tensor(self._blade_degrees)
        self._num_blades = len(self._blades)
        self._max_degree = self._blade_degrees.max()

        # [Blades, Blades, Blades]
        _list = get_cayley_tensor(self.metric, self._bases, self._blades)
        # print(_list)
        if type(_list) in [list,tuple]:
            _list = np.array(_list)
        self._cayley, self._cayley_inner, self._cayley_outer = torch.tensor(
            _list,
            dtype=torch.float32
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
            self._dual_blade_signs, dtype=torch.float32)

    def print(self, *args, **kwargs):
        """Same as the default `print` function but formats `torch.Tensor`
        instances that have as many elements on their last axis
        as the algebra has blades using `mv_repr()`.
        """
        def _is_mv(arg):
            return isinstance(arg, torch.Tensor) and len(arg.shape) > 0 and arg.shape[-1] == self.num_blades
        new_args = [self.mv_repr(arg) if _is_mv(arg) else arg for arg in args]

        print(*new_args, **kwargs)

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
    def max_degree(self) -> int:
        """Highest blade degree in the algebra."""
        return self._max_degree

    @property
    def basis_mvs(self) -> torch.Tensor:
        """List of basis vectors as torch.Tensor."""
        return self._basis_mvs

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
        tensor = tensor.to(dtype=torch.float32)
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
        tensor = tensor.to(dtype=torch.float32)
        inverted_kind_indices = self.get_kind_blade_indices(kind, invert=True)
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
        return (tensor[inverted_kind_indices]==0).sum(dim=-1)

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
        tensor = tensor.to(dtype=torch.float32)
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
            print(f"_shape={_shape},_shape_final={_shape_final}")
            print(f"i.shape={i.shape},v.shape={v.shape},b.shape={b.shape}")
            print(f"i={i},v={v},b={b}")
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
        tensor = tensor.to(dtype=torch.float32)
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

        x = blade_signs.unsqueeze(-1) *  self.blade_mvs[blade_indices]

        # a, b -> b
        return x.sum(dim=-2)        

    def __getattr__(self, name: str) -> torch.Tensor:
        """Returns basis blade tensors if name was a basis."""
        if name.startswith("e") and (name[1:] == "" or int(name[1:]) >= 0):
            return self.e(name[1:])
        raise AttributeError

    def dual(self, tensor: torch.Tensor) -> torch.Tensor:
        """Returns the dual of the geometric algebra tensor.

        Args:
            tensor: Geometric algebra tensor to return dual for

        Returns:
            Dual of the geometric algebra tensor
        """
        tensor = torch.tensor(tensor, dtype=torch.float32)
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
        tensor = tensor.to(dtype=torch.float32)
        return mv_grade_automorphism(tensor, self.blade_degrees)

    def reversion(self, tensor: torch.Tensor) -> torch.Tensor:
        """Returns the grade-reversed geometric algebra tensor.
        See https://en.wikipedia.org/wiki/Paravector#Reversion_conjugation.

        Args:
            tensor: Geometric algebra tensor to return grade-reversion for

        Returns:
            Grade-reversed geometric algebra tensor
        """
        tensor = tensor.to(dtype=torch.float32)

        return mv_reversion(tensor, self.blade_degrees)

    def conjugation(self, tensor: torch.Tensor) -> torch.Tensor:
        """Combines reversion and grade automorphism.
        See https://en.wikipedia.org/wiki/Paravector#Clifford_conjugation.

        Args:
            tensor: Geometric algebra tensor to return conjugate for

        Returns:
            Geometric algebra tensor after `reversion()` and `grade_automorphism()`
        """
        tensor = tensor.to(dtype=torch.float32)
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
        a = a.to(dtype=torch.float32)


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
        a = torch.tensor(a, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)

        return self.dual(self.ext_prod(self.dual(a), self.dual(b)))

    def ext_prod(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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
        a = a.to(dtype=torch.float32)
        b = b.to(dtype=torch.float32)

        return mv_multiply(a, b, self._cayley_outer)

    def geom_prod(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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

        a = a.to(dtype=torch.float32)
        b = b.to(dtype=torch.float32)
        return mv_multiply(a, b, self._cayley)

    
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

        a = a.to(dtype=torch.float32)
        b = b.to(dtype=torch.float32)
        return mv_multiply_element_wise(a, b, self._cayley)


    def inner_prod(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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
        a = a.to(dtype=torch.float32)
        b = b.to(dtype=torch.float32)

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
        a = a.to(dtype=torch.float32)
        k = k.to(dtype=torch.float32)

        # return mv_conv1d(a, k, self._cayley, stride=stride, padding=padding)
        return f_mv_conv1d(a, k, self._cayley, stride=stride, padding=padding)

    def mv_repr(self, a: torch.Tensor) -> str:
        """Returns a string representation for the given
        geometric algebra tensor.

        Args:
            a: Geometric algebra tensor to return the representation for

        Returns:
            string representation for `a`
        """
        a = a.to(dtype=torch.float32)


        if len(a.shape) == 1:
            return "MultiVector[%s]" % " + ".join(
                "%.2f*%s" % (value, get_blade_repr(blade_name))
                for value, blade_name
                in zip(a, self.blades)
                if value != 0
            )
        else:
            return f"MultiVector[batch_shape={a.shape[:-1]}]"

    def approx_exp(self, a: torch.Tensor, order: int = 50) -> torch.Tensor:
        """Returns an approximation of the exponential using a centered taylor series.

        Args:
            a: Geometric algebra tensor to return exponential for
            order: order of the approximation

        Returns:
            Approximation of `exp(a)`
        """
        a = a.to(dtype=torch.float32)

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
        a = a.to(dtype=torch.float32)

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
        a = a.to(dtype=torch.float32)


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
        a = a.to(dtype=torch.float32)
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
        a = a.to(dtype=torch.float32)        
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
        a = a.to(dtype=torch.float32)

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

    def inverse(self, a: torch.Tensor) -> torch.Tensor:
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
        a = a.to(dtype=torch.float32)
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
        if not torch.all(self.is_pure_kind(u, BladeKind.SCALAR)):
            raise Exception(
                "Can't invert multi-vector (det U not scalar: %s)." % u)

        # adj / det
        return u_minus_c / u[..., :1]

    def __call__(self, a: torch.Tensor) -> MultiVector:
        """Creates a `MultiVector` from a geometric algebra tensor.
        Mainly used as a wrapper for the algebra's functions for convenience.

        Args:
            a: Geometric algebra tensor to return `MultiVector` for

        Returns:
            `MultiVector` for `a`
        """
        a = a.to(dtype=torch.float32)
        return MultiVector(a, self)
        # return MultiVector(torch.tensor(a), self)
