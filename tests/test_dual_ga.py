import unittest as ut
import torch
from torch_ga import GeometricAlgebra


dual_metric = [0]
dual_bases = ["0"]
dual_blades = ["", "0"]
dual_blade_degrees = [len(blade) for blade in dual_blades]

from torch.testing import assert_close

class TestDualGeometricAlgebraMultiply(ut.TestCase):
    def assertTensorsEqual(self, a, b):
        print(f"assertTensorsEqual(a={a},b={b})")
        # print(f"a={a},b={b}")
        assert_close(a,b)

        # self.assertTrue(
        #     # tf.reduce_all(a == b), 
        #     assert_close(a,b),
        #     "%s not equal to %s" % (a, b))

    def test_mul_mv_mv(self):
        ga = GeometricAlgebra(metric=dual_metric)

        zero = ga.from_scalar(0.0)
        one = ga.from_scalar(1.0)
        eps = ga.from_tensor_with_kind(torch.ones(1), kind="pseudoscalar").squeeze()
        ten = ga.from_scalar(10.0)

        self.assertTensorsEqual(ga.geom_prod(eps, eps), zero)
        self.assertTensorsEqual(ga.geom_prod(one, one), one)
        self.assertTensorsEqual(ga.geom_prod(zero, one), zero)
        self.assertTensorsEqual(ga.geom_prod(one, zero), zero)
        self.assertTensorsEqual(ga.geom_prod(one, eps), eps)
        self.assertTensorsEqual(ga.geom_prod(eps, one), eps)
        self.assertTensorsEqual(ga.geom_prod(zero, zero), zero)
        self.assertTensorsEqual(ga.geom_prod(ten, zero), zero)
        self.assertTensorsEqual(ga.geom_prod(zero, ten), zero)
        self.assertTensorsEqual(
            ga.geom_prod(ga.geom_prod(ten, eps), eps),
            zero
        )
        self.assertTensorsEqual(ga.geom_prod(ten, one), ten)
        self.assertTensorsEqual(ga.geom_prod(one, ten), ten)

    def test_mul_tf_mv(self):
        ga = GeometricAlgebra(metric=dual_metric)

        zero = ga.from_scalar(0.0)
        one = ga.from_scalar(1.0)

        # eps = ga.from_tensor_with_kind(torch.ones(1), kind="pseudoscalar")
        eps = ga.from_tensor_with_kind(torch.tensor([1]), kind="pseudoscalar").squeeze()
        ten = ga.from_scalar(10.0)
        if False:        
            print(f"one={one}")
            print(f"zero={zero}")
            print(f"eps={eps}")
            print(f"ten={ten}")

        zero_tf = torch.tensor([0, 0], dtype=torch.float32)
        one_tf = torch.tensor([1, 0], dtype=torch.float32)
        eps_tf = torch.tensor([0, 1], dtype=torch.float32)
        ten_tf = torch.tensor([10, 0], dtype=torch.float32)
        if False:        
            print(f"zero_tf={zero_tf}")
            print(f"one_tf={one_tf}")
            print(f"eps_tf={eps_tf}")
            print(f"ten_tf={ten_tf}")

            print(f"ga.geom_prod(one, one_tf)={ga.geom_prod(one, one_tf)}")

        self.assertTensorsEqual(ga.geom_prod(one, one_tf), one)
        self.assertTensorsEqual(ga.geom_prod(one_tf, one), one)
        self.assertTensorsEqual(ga.geom_prod(zero, one_tf), zero)
        self.assertTensorsEqual(ga.geom_prod(one_tf, zero), zero)
        self.assertTensorsEqual(ga.geom_prod(zero_tf, one), zero)
        self.assertTensorsEqual(ga.geom_prod(one, zero_tf), zero)
        self.assertTensorsEqual(ga.geom_prod(one_tf, eps), eps)
        self.assertTensorsEqual(ga.geom_prod(eps, one_tf), eps)
        self.assertTensorsEqual(ga.geom_prod(zero_tf, zero), zero)
        self.assertTensorsEqual(ga.geom_prod(zero, zero_tf), zero)
        self.assertTensorsEqual(ga.geom_prod(ten_tf, zero), zero)
        self.assertTensorsEqual(ga.geom_prod(zero, ten_tf), zero)
        self.assertTensorsEqual(ga.geom_prod(ten, zero_tf), zero)
        self.assertTensorsEqual(ga.geom_prod(zero_tf, ten), zero)
        self.assertTensorsEqual(
            ga.geom_prod(ga.geom_prod(ten_tf, eps), eps),
            zero
        )
        self.assertTensorsEqual(ga.geom_prod(ten_tf, one), ten)
        self.assertTensorsEqual(ga.geom_prod(one, ten_tf), ten)
        self.assertTensorsEqual(ga.geom_prod(ten, one_tf), ten)
        self.assertTensorsEqual(ga.geom_prod(one_tf, ten), ten)


class TestDualGeometricAlgebraMisc(ut.TestCase):
    def assertTensorsEqual(self, a, b):
        # self.assertTrue(tf.reduce_all(a == b), "%s not equal to %s" % (a, b))
        self.assertTrue(torch.all(a == b), "%s not equal to %s" % (a, b))

    def test_auto_diff_square(self):
        """Test automatic differentiation using
        dual numbers for the square function.
        f(x) = x^2
        f'(x) = d/dx f(x) = 2x
        """
        ga = GeometricAlgebra(metric=dual_metric)

        one = ga.from_scalar(1.0)
        five = ga.from_scalar(5.0)
        eps = ga.from_tensor_with_kind(torch.ones(1), kind="pseudoscalar")

        x = one + eps
        if False:        
            print(f"one={one}")
            print(f"eps={eps}")
            print(f"x = one + eps={x}")        

        # f(1) = 1^2 = 1, f'(1) = 2
        x_squared = ga.geom_prod(x, x)
        self.assertTensorsEqual(ga.select_blades_with_name(x_squared, ""), 1.0)
        self.assertTensorsEqual(ga.select_blades_with_name(x_squared, "0"), 2.0)

        y = five + eps

        # f(5) = 5^2 = 25, f'(5) = 10
        y_squared = ga.geom_prod(y, y)
        self.assertTensorsEqual(ga.select_blades_with_name(y_squared, ""), 25.0)
        self.assertTensorsEqual(ga.select_blades_with_name(y_squared, "0"), 10.0)

    def test_batched_auto_diff_square(self):
        """Test automatic differentiation using
        dual numbers for the square function.
        Use batch with identical elements.
        f(x) = x^2
        f'(x) = d/dx f(x) = 2x
        """
        ga = GeometricAlgebra(metric=dual_metric)

        one = ga.from_tensor_with_kind(torch.ones([3, 4, 1]), kind="scalar")
        five = ga.from_tensor_with_kind(torch.full([3, 4, 1], 5.0), kind="scalar")
        eps = ga.from_tensor_with_kind(torch.ones([3, 4, 1]), kind="pseudoscalar")
        if False:        
            print(f"five={five},one={one}")
            print(f"five.shape={five.shape},one.shape={one.shape}")

        x = one + eps
        if False:        
            print(f"one={one}")
            print(f"eps={eps}")
            print(f"x = one + eps={x}")

        # f(1) = 1^2 = 1, f'(1) = 2
        x_squared = ga.geom_prod(x, x)

        if False:        
            print(f"x_squared={x_squared}")
            print(f"x_squared.shape={x_squared.shape}")
            _a = ga.select_blades_with_name(x_squared, '')
            _b = ga.select_blades_with_name(x_squared, '0')
            print(f"ga.select_blades_with_name(x_squared, '')={_a}")
            print(f"_a.shape={_a.shape}")
            print(f"ga.select_blades_with_name(x_squared, '0')={_b}")
            print(f"_b.shape={_b.shape}")

        self.assertTensorsEqual(ga.select_blades_with_name(x_squared, ""), 1.0)
        self.assertTensorsEqual(ga.select_blades_with_name(x_squared, "0"), 2.0)

        y = five + eps

        # f(5) = 5^2 = 25, f'(5) = 10
        y_squared = ga.geom_prod(y, y)
        self.assertTensorsEqual(ga.select_blades_with_name(y_squared, ""), 25.0)
        self.assertTensorsEqual(ga.select_blades_with_name(y_squared, "0"), 10.0)

    def test_mul_inverse(self):
        ga = GeometricAlgebra(metric=dual_metric)

        # a = 2
        a = ga.from_tensor_with_kind(torch.full([1], 2.0), kind="scalar")
        if False: print(f"a={a},torch.full([1], 2.0)={torch.full([1], 2.0)}")


        # b = 3 + 3e0
        b = ga.from_tensor_with_kind(torch.full([2], 3.0), kind="mv")

        # a * b = 2 * (3 + 3e0) = 6 + 6e0
        c = ga.geom_prod(a, b)
        self.assertTensorsEqual(c, ga.from_scalar(6.0) + 6.0 * ga.e("0"))
        if False:
            print(f"a={a}")
            print(f"b={b}")
            print(f"c=a*b={c}")

        # a^-1 = 1 / 2
        a_inv = ga.inverse(a)
        self.assertTensorsEqual(ga.select_blades_with_name(a_inv, ""), 0.5)
        if False:
            print(f"a={a}")
            print(f"a_inv={a_inv}")

        # c = a * b
        # => a_inv * c = b
        self.assertTensorsEqual(ga.geom_prod(a_inv, c), b)
        if False:
            print(f"b={b}")
            print(f"a_inv * c={ga.geom_prod(a_inv, c)}")

        # Since a is scalar, should commute too.
        # => c * a_inv = b
        self.assertTensorsEqual(ga.geom_prod(c, a_inv), b)
        if False:
            print(f"b={b}")
            print(f"a_inv * c={ga.geom_prod(c, a_inv)}")

        # b is not simply invertible (because it does not square to a scalar)
        # and will throw an exception
        self.assertRaises(Exception, ga.simple_inverse, b)

        # b is invertible with the shirokov inverse
        if False: print(f"b={b}")
        b_inv = ga.inverse(b)
        if False: print(f"b_inv={b_inv}")
        self.assertTensorsEqual(ga.geom_prod(b, b_inv), 1 * ga.e(""))


if __name__ == '__main__':
    ut.main()        