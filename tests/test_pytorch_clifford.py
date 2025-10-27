from torch_ga.clifford import CliffordAlgebra
import unittest as ut
import torch


from torch.testing import assert_close

class TestDualGeometricAlgebraMultiply(ut.TestCase):
    
    def assertTensorsApproxEqual(self, a, b, tolerance=1e-4):
        print(f"assertTensorsApproxEqual(a={a},b={b})")
        assert_close(a,b,atol=tolerance, rtol=0), "%s not equal to %s" % (a, b)
        
    def test_prod(self):
        pga_signature = [0, 1, 1, 1]
        pga_signature = [1, 1, -1]
        pga = CliffordAlgebra(pga_signature)

        # a = 3e01 + 5e02
        a = pga.embed(torch.tensor([3,5]),(3,4))
        b = pga.embed(torch.tensor([2,1]),(3,4))
                      
        self.assertTensorsApproxEqual(pga.product(a,b),pga.inner_product(a,b)+pga.outer_product(a,b))
        
        a = pga(a)
        b = pga(b)
        print(a,b)

if __name__ == '__main__':
    ut.main()        