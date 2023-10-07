# torch_ga - PyTorch Geometric Algebra
[Build status](https://github.com/Falesiani/torch_ga/actions)

[GitHub](https://github.com/falesiani/torch_ga) | [Docs](https://github.com/falesiani/torch_ga) | [Benchmarks](https://github.com/falesiani/torch_ga/tree/master/benchmarks) 

Python package for Geometric / Clifford Algebra with Pytorch.

**This project is a work in progress. Its API may change and the examples aren't polished yet.**

**This project is based on the TF-GA library [TGA](https://doi.org/10.5281/zenodo.3902404)**


Pull requests and suggestions either by opening an issue or by [sending me an email](mailto:francesco.alesiani@neclab.eu) are welcome.

## Installation
Install using pip: `pip install torch_ga`

Requirements:
- Python 3
- torch
- numpy

### Conda Environment
An example environment is provided, but please feel free to create your own custom environment
```
conda create -n torch_ga -f environment.yml
```

## Basic usage
There are two ways to use this library. In both ways we first create a [`GeometricAlgebra`]() instance given a metric.
Then we can either work on [`torch.Tensor`]() instances directly where the last axis is assumed to correspond to
the algebra's blades.
```python
import torch
from torch_ga import GeometricAlgebra

# Create an algebra with 3 basis vectors given their metric.
# Contains geometric algebra operations.
ga = GeometricAlgebra(metric=[1, 1, 1])

# Create geometric algebra torch.Tensor for vector blades (ie. e_0 + e_1 + e_2).
# Represented as torch.Tensor with shape [8] (one value for each blade of the algebra).
# torch.Tensor: [0, 1, 1, 1, 0, 0, 0, 0]
ordinary_vector = ga.from_tensor_with_kind(torch.ones(3), kind="vector")

# 5 + 5 e_01 + 5 e_02 + 5 e_12
quaternion = ga.from_tensor_with_kind(torch.fill(dims=4, value=5), kind="even")

# 5 + 1 e_0 + 1 e_1 + 1 e_2 + 5 e_01 + 5 e_02 + 5 e_12
multivector = ordinary_vector + quaternion

# Inner product e_0 | (e_0 + e_1 + e_2) = 1
# ga.print is like print, but has extra formatting for geometric algebra torch.Tensor instances.
ga.print(ga.inner_prod(ga.e0, ordinary_vector))

# Exterior product e_0 ^ e_1 = e_01.
ga.print(ga.ext_prod(ga.e0, ga.e1))

# Grade reversal ~(5 + 5 e_01 + 5 e_02 + 5 e_12)
# = 5 + 5 e_10 + 5 e_20 + 5 e_21
# = 5 - 5 e_01 - 5 e_02 - 5 e_12
ga.print(ga.reversion(quaternion))

# torch.Tensor 5
ga.print(quaternion[0])

# torch.Tensor of shape [1]: -5 (ie. reversed sign of e_01 component)
ga.print(ga.select_blades_with_name(quaternion, "10"))

# torch.Tensor of shape [8] with only e_01 component equal to 5
ga.print(ga.keep_blades_with_name(quaternion, "10"))
```

Alternatively we can convert the geometric algebra [`torch.Tensor`]() instance to [`MultiVector`]()
instances which wrap the operations and provide operator overrides for convenience.
This can be done by using the `__call__` operator of the [`GeometricAlgebra`]() instance.
```python
# Create geometric algebra torch.Tensor instances
a = ga.e123
b = ga.e1

# Wrap them as `MultiVector` instances
mv_a = ga(a)
mv_b = ga(b)

# Reversion ((~mv_a).tensor equivalent to ga.reversion(a))
print(~mv_a)

# Geometric / inner / exterior product
print(mv_a * mv_b)
print(mv_a | mv_b)
print(mv_a ^ mv_b)
```

## Keras layers
torch_ga also provides [Keras-like]() layers which provide
layers similar to the existing ones but using multivectors instead. For example the [`GeometricProductDense`]()
layer is exactly the same as the [`Dense`]() layer but uses
multivector-valued weights and biases instead of scalar ones. The exact kind of multivector-type can be
passed too. Example:

```python
import torch as tf
from torch_ga import GeometricAlgebra
from torch_ga.layers import TensorToGeometric, GeometricToTensor, GeometricProductDense

# 4 basis vectors (e0^2=+1, e1^2=-1, e2^2=-1, e3^2=-1)
sta = GeometricAlgebra([1, -1, -1, -1])

# We want our dense layer to perform a matrix multiply
# with a matrix that has vector-valued entries.
vector_blade_indices = sta.get_kind_blade_indices(BladeKind.VECTOR),

# Create our input of shape [Batch, Units, BladeValues]
tensor = torch.ones([20, 6, 4])

# The matrix-multiply will perform vector * vector
# so our result will be scalar + bivector.
# Use the resulting blade type for the bias too which is
# added to the result.
result_indices = torch.concat([
    sta.get_kind_blade_indices(BladeKind.SCALAR), # 1 index
    sta.get_kind_blade_indices(BladeKind.BIVECTOR) # 6 indices
], axis=0)

sequence = nn.Sequential([
    # Converts the last axis to a dense multivector
    # (so, 4 -> 16 (total number of blades in the algebra))
    TensorToGeometric(sta, blade_indices=vector_blade_indices),
    # Perform matrix multiply with vector-valued matrix
    GeometricProductDense(
        algebra=sta, units=8, # units is analagous to Keras' Dense layer
        blade_indices_kernel=vector_blade_indices,
        blade_indices_bias=result_indices
    ),
    # Extract our wanted blade indices (last axis 16 -> 7 (1+6))
    GeometricToTensor(sta, blade_indices=result_indices)
])

# Result will have shape [20, 8, 7]
result = sequence(tensor)
```


### Available layers
| Class | Description |
|--|--|
| [`GeometricProductDense`] | Analagous to Keras' [`Dense`] with multivector-valued weights and biases. Each term in the matrix multiplication does the geometric product `x * w`. |
| [`GeometricSandwichProductDense`] | Analagous to Keras' [`Dense`] with multivector-valued weights and biases. Each term in the matrix multiplication does the geometric product `w *x * ~w`. |
| [`GeometricProductElementwise`] | Performs multivector-valued elementwise geometric product of the input units with a different weight for each unit. |
| [`GeometricSandwichProductElementwise`] | Performs multivector-valued elementwise geometric sandwich product of the input units with a different weight for each unit. |
| [`GeometricProductConv1D`] | Analagous to Keras' [`Conv1D`] with multivector-valued kernels and biases. Each term in the kernel multiplication does the geometric product `x * k`. |
| [`TensorToGeometric`] | Converts from a [`torch.Tensor`] to the geometric algebra [`torch.Tensor`] with as many blades on the last axis as basis blades in the algebra where blade indices determine which basis blades the input's values belong to. |
| [`GeometricToTensor`] | Converts from a geometric algebra [`torch.Tensor`] with as many blades on the last axis as basis blades in the algebra to a [`torch.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) where blade indices determine which basis blades we extract for the output. |
| [`TensorWithKindToGeometric`] | Same as [`TensorToGeometric`] but using [`BladeKind`] (eg. `"bivector"`, `"even"`) instead of blade indices. |
| [`GeometricToTensorWithKind`] | Same as [`GeometricToTensor`] but using [`BladeKind`] (eg. `"bivector"`, `"even"`) instead of blade indices. |
| [`GeometricAlgebraExp`]| Calculates the exponential function of the input. Input must square to a scalar. |


## Notebooks
[Generic examples](https://github.com/Falesiani/torch_ga/tree/master/notebooks/torch_ga.ipynb)

[Using Keras layers to estimate triangle area](https://github.com/Falesiani/torch_ga/tree/master/notebooks/keras-triangles.ipynb)

[Classical Electromagnetism using Geometric Algebra](https://github.com/Falesiani/torch_ga/tree/master/notebooks/em.ipynb)

[Quantum Electrodynamics using Geometric Algebra](https://github.com/Falesiani/torch_ga/tree/master/notebooks/qed.ipynb)

[Projective Geometric Algebra](https://github.com/Falesiani/torch_ga/tree/master/notebooks/pga.ipynb)

[1D Multivector-valued Convolution Example](https://github.com/Falesiani/torch_ga/tree/master/notebooks/conv.ipynb)

## Tests
Tests using Python's built-in [`unittest`](https://docs.python.org/3/library/unittest.html) module are available in the `tests` directory. All tests can be run by
executing `python -m unittest discover tests` from the root directory of the repository.

## Citing
For citing all versions the following BibTeX can be used

```
@software{torch_ga,
  author       = {Alesiani, Francesco},
  title        = {PyTorch Geometric Algebra},
  publisher    = {Github},
  url          = {https://github.com/falesiani/torch_ga}
}
```

## Disclaimer
PyTorch, the PyTorch logo and any related marks are trademarks of Facebook, Inc.
