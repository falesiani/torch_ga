"""based on https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks"""

import functools
import math

import torch
from torch import nn

from torch_ga.clifford.blades import ShortLexBasisBladeOrder, construct_gmt, gmt_element
from torch_ga import GeometricAlgebra, MultiVector

from icecream import ic

tdet = lambda x: torch.as_tensor(x.detach()) if isinstance(x, torch.Tensor) else x
totorch = lambda x: torch.as_tensor(x) if isinstance(x,(list,tuple)) else x
# topar = lambda x: torch.nn.Parameter(tdet(totorch(x)), requires_grad=False)
topar = lambda x: torch.nn.Parameter(totorch(x), requires_grad=False)
ltopar = lambda x: [ topar(_) for _ in x]
class CliffordAlgebra(nn.Module):            
    def __init__(self, metric, device='cpu'):
        super().__init__()

        # self.register_buffer("metric", torch.as_tensor(metric.detach() if isinstance(metric, torch.Tensor) else metric))
        self.metric = topar(metric)
        # self.num_bases = len(metric)
        self.n_bases = len(metric)
        self._p,self._q,self._r = sum([1 for _ in metric if _>0]),sum([1 for _ in metric if _<0]),sum([1 for _ in metric if _==0])
        self.bbo = ShortLexBasisBladeOrder(self.num_bases)
        self._dim = len(self.metric)
        # self.num_blades = 
        self.n_blades = len(self.bbo.grades)
        cayley, cayley_inner, cayley_outer  = [_ for _ in construct_gmt(self.bbo.index_to_bitmap, self.bbo.bitmap_to_index, self.metric)]
        cayley, cayley_inner, cayley_outer  = [ _.to_dense().to(torch.get_default_dtype()) for _ in [cayley, cayley_inner, cayley_outer]] 
        self.grades = topar(self.bbo.grades.unique())
        self.subspaces =  topar(torch.tensor(tuple(math.comb(self.dim, g) for g in self.grades)))
        # self.register_buffer(
        #     "subspaces",
        #     torch.tensor(tuple(math.comb(self.dim, g) for g in self.grades)),
        # )
        self.n_subspaces = len(self.grades)
        self.grade_to_slice = self._grade_to_slice(self.subspaces)
        self.grade_to_index = [
            topar(torch.tensor(range(*s.indices(s.stop)))) for s in self.grade_to_slice
        ]

        # self.register_buffer(
        #     "bbo_grades", self.bbo.grades.to(torch.get_default_dtype())
        # )
        # self.register_buffer("even_grades", self.bbo_grades % 2 == 0)
        # self.register_buffer("odd_grades", ~self.even_grades)
        # self.register_buffer("cayley", cayley)
        # self.register_buffer("cayley_inner", cayley_inner)
        # self.register_buffer("cayley_outer", cayley_outer)
        self.bbo_grades = topar(self.bbo.grades.to(torch.get_default_dtype()))
        
        self.even_grades=topar(self.bbo_grades % 2 == 0)
        self.odd_grades=topar(~self.even_grades)
        self.cayley=topar(cayley)
        self.cayley_inner=topar(cayley_inner)
        self.cayley_outer=topar(cayley_outer)
        
        self.to(device)
        
    # def to(self,device):
    #     # super().to(device)
    #     self.device=device
    #     self.metric=self.metric.to(device)
    #     self.grades=self.grades.to(device)
    #     self.subspaces=self.subspaces.to(device)
    #     # self.grade_to_slice.to(device)
    #     self.grade_to_index = [_.to(device) for _ in self.grade_to_index]

    #     self.bbo_grades = self.bbo_grades.to(device)
        
    #     self.even_grades=self.even_grades.to(device)
    #     self.odd_grades=self.odd_grades.to(device)
    #     self.cayley=self.cayley.to(device)
    #     self.cayley_inner=self.cayley_inner.to(device)
    #     self.cayley_outer=self.cayley_outer.to(device)
        
    #     return self
        
    # def register_buffer(self, name, tensor, persistent = True):        
        # return super().register_buffer(name, tensor, persistent)
    
    def geometric_product(self, a, b, blades=None):
        return self.product(a, b, blades, _type="geometric")
    def inner_product(self, a, b, blades=None):
        return self.product(a, b, blades, _type="inner")
    def outer_product(self, a, b, blades=None):
        return self.product(a, b, blades, _type="outer")
    def product(self, a, b, blades=None, _type="geometric"):
        if _type in ["geometric"]:
            cayley = self.cayley
        elif _type in ["inner"]:
            cayley = self.cayley_inner
        elif _type in ["outer"]:
            cayley = self.cayley_outer
        else:
            raise Exception(f"Unknownd product {_type}")

        if blades is not None:
            blades_l, blades_o, blades_r = blades
            assert isinstance(blades_l, torch.Tensor)
            assert isinstance(blades_o, torch.Tensor)
            assert isinstance(blades_r, torch.Tensor)
            cayley = cayley[blades_l[:, None, None].long(), blades_o[:, None].long(), blades_r.long()]
                    
        # return torch.einsum("...i,ijk,...k->...j", a, cayley, b)
        # return torch.einsum("...i,...j,ikj->...k", a, b, cayley) # equivalent GL
        return torch.einsum("...i,...j,ijk->...k", a, b, cayley) #from GA
        

    def __call__(self, a: torch.Tensor) -> MultiVector:
        """Creates a `MultiVector` from a geometric algebra tensor.
        Mainly used as a wrapper for the algebra's functions for convenience.

        Args:
            a: Geometric algebra tensor to return `MultiVector` for

        Returns:
            `MultiVector` for `a`
        """
        if False: a = a.to(dtype=torch.float32)
        # return MultiVector(a, GeometricAlgebra(self.metric.detach().numpy()))
        # return MultiVector(a, GeometricAlgebra(self.metric, device=self.cayley.device))
        # ga = GeometricAlgebra(self.metric,device=a.device if device is None else device)
        ga = GeometricAlgebra(self.metric)
        ga = ga.to(a.device)
        # ic(a.device, ga.device)
        return MultiVector(a,ga)
        # return MultiVector(a, self)
    
    def to_ga(self) -> GeometricAlgebra:
        """Creates a `GeometricAlgebra` from the Clifford Algebra

        Returns:
            `GeometricAlgebra` 
        """
        return GeometricAlgebra(self.metric)
    
    def _grade_to_slice(self, subspaces):
        grade_to_slice = list()
        subspaces = torch.as_tensor(subspaces)
        for grade in self.grades:
            index_start = subspaces[:grade].sum()
            index_end = index_start + math.comb(self.dim, grade)
            grade_to_slice.append(slice(index_start, index_end))
        return grade_to_slice

    @functools.cached_property
    def _alpha_signs(self):
        return torch.pow(-1, self.bbo_grades)

    @functools.cached_property
    def _beta_signs(self):
        return torch.pow(-1, self.bbo_grades * (self.bbo_grades - 1) // 2)

    @functools.cached_property
    def _gamma_signs(self):
        return torch.pow(-1, self.bbo_grades * (self.bbo_grades + 1) // 2)

    @property
    def sign_p(self) -> int:
        """ G(p,q,r)
        """
        return self._p
    @property
    def sign_q(self) -> int:
        """ G(p,q,r)
        """
        return self._q
    @property
    def sign_r(self) -> int:
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
        return self.n_bases
    @property
    def num_blades(self) -> int:
        """ number of basis degree=1
        """
        return self.n_blades
    @property
    def blade_degrees(self):
        return self.bbo_grades

    def grade_automorphism(self, x):        
        signs = 1.0 - 2.0 * (self.blade_degrees % 2.0)
        return signs * x
    
    def conjugation(self, x):
        return self.grade_automorphism(self.reverse(x))
    def conjugate(self, x):
        return self.grade_automorphism(self.reverse(x))
    
    def alpha(self, mv, blades=None):
        """Clifford main involution"""
        
        signs = self._alpha_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def beta(self, mv, blades=None):
        """Clifford main anti-involution (reversion)"""
        signs = self._beta_signs
        if blades is not None:
            signs = signs[blades.long()]
        return signs * mv.clone()

    def gamma(self, mv, blades=None):
        """Clifford conjugation"""
        signs = self._gamma_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def zeta(self, mv):
        return mv[..., :1]

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
    
    def get_scalar(self,a):
        """scalar part of multivector."""
        return a[...,0]
    def get_pseudo_scalar(self,a):
        """scalar part of multivector."""
        return a[...,-1]
    def get_vector(self,a):
        """vector part of multivector."""
        return a[...,1:self.dim+1]
    def get_pseudo_vector(self,a):
        """vector part of multivector."""
        return a[...,-self.dim-1:-1]
    def get_bivector(self,a):
        """bivector part of multivector."""        
        s = self.grade_to_slice[2]
        return a[..., s]
    def get_pseudo_bivector(self,a):
        """pseudo bivector part of multivector."""
        s = self.grade_to_slice[self.dim-2]
        return a[..., s]
    def get_trivector(self,a):
        """bivector part of multivector."""        
        s = self.grade_to_slice[3]
        return a[..., s]
    def get_pseudo_trivector(self,a):
        """pseudo bivector part of multivector."""
        s = self.grade_to_slice[self.dim-3]
        return a[..., s]
    def get_blade(self,a,deg):
        """deg-degreee blade part of multivector."""
        s = self.grade_to_slice[deg]
        return a[..., s]

    def b(self, x, y, blades=None):
        """Bilinear Form b(x,y): xy+yx = 2b(x,y), I would call the inner product

        Args:
            x (_type_): _description_
            y (_type_): _description_
            blades (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if blades is not None:
            assert len(blades) == 2
            beta_blades = blades[0]
            blades = (
                blades[0],
                torch.tensor([0]),
                blades[1],
            )
        else:
            blades = torch.tensor(range(self.n_blades),dtype=torch.long).to(self.cayley)
            blades = (
                blades,
                torch.tensor([0],dtype=torch.long).to(self.cayley),
                blades,
            )
            beta_blades = None

        return self.geometric_product(
            self.beta(x, blades=beta_blades),
            y,
            blades=blades,
        )

    def q(self, mv, blades=None):
        """Scalar Square b(x), defines the bilinear form 2 b(x,y) = q(x+y)- q(x) -q(y)

        Args:
            mv (tensor): _description_
            blades (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if blades is not None:
            blades = (blades, blades)
        return self.b(mv, mv, blades=blades)

    def _smooth_abs_sqrt(self, input, eps=1e-16):
        return (input**2 + eps) ** 0.25

    def norm_(self, mv, blades=None):
        return self._smooth_abs_sqrt(self.q(mv, blades=blades))
    
    def norm(self, mv, blades=None):
        return self._smooth_abs_sqrt(self.product(mv, self.conjugate(mv), blades=blades)[...,0])
        # return abs(self.product(mv, self.conjugate(mv), blades=blades)[...,0])**0.5

    def norms(self, mv, grades=None):
        if grades is None:
            grades = self.grades
        return [
            self.norm(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in grades
        ]

    def qs(self, mv, grades=None):
        if grades is None:
            grades = self.grades
        return [
            self.q(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in grades
        ]

    def reverse(self, mv, blades=None):
        """Perform the reversion operation on multivectors, an operation specific to geometric algebra.

        In Geometric Algebra, the reverse of a multivector is formed by reversing the order of the vectors in each blade.

        Args:
            mv (torch.Tensor): Input multivectors.
            blades (Union[tuple, list, torch.Tensor], optional): Specify which blades are present in the multivector.

        Returns:
            torch.Tensor: The reversed multivector.
        """
        grades = self.bbo.grades.to(mv.device)
        if blades is not None:
            grades = grades[torch.as_tensor(blades, dtype=int)]
        signs = torch.pow(-1, torch.floor(grades * (grades - 1) / 2))
        return signs * mv.clone()
    
    def sandwich(self, u, v, w=None):
        if w is None:
            return self.sandwich2(u,v)
        else:
            return self.sandwich3(u,v,w)
                
    def sandwich2(self, a, b):
        """aba'"""
        return self.geometric_product(self.geometric_product(a, b), self.reverse(a))
    
    def sandwich3(self, u, v, w):
        return self.geometric_product(self.geometric_product(u, v), w)

    def output_blades(self, blades_left, blades_right):
        blades = []
        for blade_left in blades_left:
            for blade_right in blades_right:
                bitmap_left = self.bbo.index_to_bitmap[blade_left]
                bitmap_right = self.bbo.index_to_bitmap[blade_right]
                bitmap_out, _ = gmt_element(bitmap_left, bitmap_right, self.metric)
                index_out = self.bbo.bitmap_to_index[bitmap_out]
                blades.append(index_out)

        return torch.tensor(blades)

    def random(self, n=None):
        if n is None:
            n = 1
        return torch.randn(n, self.n_blades)

    def random_vector(self, n=None):
        if n is None:
            n = 1
        vector_indices = self.bbo_grades == 1
        v = torch.zeros(n, self.n_blades, device=self.cayley.device)
        v[:, vector_indices] = torch.randn(
            n, vector_indices.sum(), device=self.cayley.device
        )
        return v

    def parity(self, mv):
        is_odd = torch.all(mv[..., self.even_grades] == 0)
        is_even = torch.all(mv[..., self.odd_grades] == 0)

        if is_odd ^ is_even:  # exclusive or (xor)
            return is_odd
        else:
            raise ValueError("This is not a homogeneous element.")

    def eta(self, w):
        """Coboundary of alpha"""
        return (-1) ** self.parity(w)

    def alpha_w(self, w, mv):
        return self.even_grades * mv + self.eta(w) * self.odd_grades * mv

    def inverse(self, mv, blades=None):
        mv_ = self.beta(mv, blades=blades)
        return mv_ / self.q(mv)

    def rho(self, w, mv):
        """Applies the versor w action to mv.
        
        Reflects x in the hyperplane normal to w
        
        rho(w)(x) = -wxw^-1 = x-e b(w,x)/b(w,w)w
        
        """
        
        return self.sandwich3(w, self.alpha_w(w, mv), self.inverse(w))

    def reduce_geometric_product(self, inputs):
        return functools.reduce(self.geometric_product, inputs)

    def versor(self, order=None, normalized=True):
        if order is None:
            order = self.dim if self.dim % 2 == 0 else self.dim - 1
        vectors = self.random_vector(order)
        versor = self.reduce_geometric_product(vectors[:, None])
        if normalized:
            versor = versor / self.norm(versor)[..., :1]
        return versor

    def rotor(self):
        return self.versor()

    @functools.cached_property
    def geometric_product_paths(self):
        gp_paths = torch.zeros((self.dim + 1, self.dim + 1, self.dim + 1), dtype=bool)

        for i in range(self.dim + 1):
            for j in range(self.dim + 1):
                for k in range(self.dim + 1):
                    s_i = self.grade_to_slice[i]
                    s_j = self.grade_to_slice[j]
                    s_k = self.grade_to_slice[k]

                    m = self.cayley[s_i, s_j, s_k]
                    gp_paths[i, j, k] = (m != 0).any()

        return gp_paths

