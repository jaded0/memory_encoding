"""Small deterministic CPU tests of the analysis, not the training library."""
import unittest
import numpy as np
from measure import analyze

class MetricsTest(unittest.TestCase):
    def test_uniform_and_disjoint(self):
        N=1024
        E=np.ones((1,2,N),dtype=float)
        A=np.zeros_like(E)
        A[0,0,:256]=1
        A[0,1,256:512]=1
        d=analyze(E,A,np.ones((1,2),bool),[[0,1]],"mqar")
        self.assertEqual(d['pr_store']['median'],1)
        for k in (16,64,256):
            e=d['selections']['energy'][str(k)]
            a=d['selections']['activation'][str(k)]
            self.assertEqual(e['store']['median'],k/N)
            self.assertEqual(e['jaccard']['mean'],1)
            self.assertEqual(a['jaccard']['mean'],0)
            self.assertEqual(a['ever_selected'],2*k/N)
            self.assertEqual(e['baseline'],k/(2*N-k))
    def test_padding_and_zero(self):
        E=np.ones((1,3,1024),dtype=float)
        E[0,0]=0
        d=analyze(E,E,np.array([[True,True,False]]),[[1]],"kr")
        self.assertEqual(d['zero_energy_steps'],1)
        self.assertEqual(d['pr_all']['n'],1)
        self.assertEqual(d['selections']['energy']['64']['jaccard']['n'],1)
    def test_energy_formula(self):
        p=np.array([2.,3.])
        x=np.array([1.,4.,2.])
        mask=np.array([[1.,0.,1.],[0.,1.,1.]])
        E=p**2*(mask@x**2)
        direct=((p[:,None]*x[None,:]*mask)**2).sum(-1)
        np.testing.assert_array_equal(E,direct)

if __name__ == '__main__':
    unittest.main()
