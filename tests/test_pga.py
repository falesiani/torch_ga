# from tfga import GeometricAlgebra
from torch_ga import GeometricAlgebra
import unittest as ut
import torch


pga_signature = [0, 1, 1, 1]

from torch.testing import assert_close
class TestDualGeometricAlgebraMultiply(ut.TestCase):
    def assertTensorsApproxEqual(self, a, b, tolerance=1e-4):
        print(f"assertTensorsApproxEqual(a={a},b={b})")
        assert_close(a,b,atol=tolerance, rtol=0), "%s not equal to %s" % (a, b)

    # def assertTensorsApproxEqual(self, a, b, tolerance=1e-4):
    #     self.assertTrue(tf.reduce_all(tf.abs(a - b) < tolerance),
    #                     "%s not equal to %s" % (a, b))

    def test_exp_eq_approx_exp_e01_e02(self):
        pga = GeometricAlgebra(pga_signature)

        # a = 3e01 + 5e02
        a = 3 * pga.e01 + 5 * pga.e02
        print("test_exp_eq_approx_exp_e01_e02")
        print(f"a={a}")
        # exp(a) = 1 + 3e01 + 5e02
        self.assertTensorsApproxEqual(pga.approx_exp(a), pga.exp(a))

    def test_exp_eq_approx_exp_e12_e23(self):
        pga = GeometricAlgebra(pga_signature)

        # a = 3e12 + 5e23
        a = 3 * pga.e12 + 5 * pga.e23
        print("test_exp_eq_approx_exp_e12_e23")
        print(f"a={a}")
        # exp(a) ~= 0.90 - 0.22e12 -0.37e23
        self.assertTensorsApproxEqual(pga.approx_exp(a), pga.exp(a))

    def test_inverse(self):
        pga = GeometricAlgebra(pga_signature)

        # a = 3e12 + 5e23
        a = 3 * pga.e12 + 5 * pga.e23

        # a_inv: -0.09*e_12 + -0.15*e_23
        a_inv = pga.inverse(a)

        # print("test_inverse")
        # print(f"a={a}")
        # print(f"a_inv={a_inv}")

        # a a_inv should be 1
        self.assertTensorsApproxEqual(pga.geom_prod(a, a_inv), 1 * pga.e(""))


if __name__ == '__main__':
    ut.main()    