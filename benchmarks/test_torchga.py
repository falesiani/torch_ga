from torch_ga import GeometricAlgebra
import torch
import pytest


def _torchga_add(a, b):
    return a + b


def _torchga_mul(ga, a, b):
    return ga.geom_prod(a, b)


@pytest.mark.parametrize("num_elements", [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000])
def test_tfga_add_mv_mv(num_elements, benchmark):
    ga = GeometricAlgebra([1, -1, -1, -1])
    a = torch.ones([num_elements, ga.num_blades])
    b = torch.ones([num_elements, ga.num_blades])
    benchmark(_torchga_add, a, b)


@pytest.mark.parametrize("num_elements", [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000])
def test_tfga_mul_mv_mv(num_elements, benchmark):
    ga = GeometricAlgebra([1, -1, -1, -1])
    a = torch.ones([num_elements, ga.num_blades])
    b = torch.ones([num_elements, ga.num_blades])
    benchmark(_torchga_mul, ga, a, b)




# if __name__ == '__main__':
#     ut.main()        