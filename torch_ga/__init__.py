"""
PyTorch Geometric Algebra

An python library for Geometric Algebra in Pytorch
"""

__author__ = 'Francesco Alesiani'
__credits__ = 'Torch-GA'
  
from .torch_ga import *
from .blades import *
from .cayley import *
from .layers import *
from .mv_ops import *
from .mv import *
from .utils import *
from .plots import *
from .jacobian import *
from .transform import *
from .clifford.algebra import CliffordAlgebra
from .__version__ import __version__
