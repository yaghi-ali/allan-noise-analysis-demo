import unittest
import numpy as np
from analysis import *

class Tests(unittest.TestCase):
    def test_constant(self):
        t,a=allan(np.ones(100),.1,[1,3,10]);np.testing.assert_allclose(a,0,atol=1e-14)
    def test_linear_ramp(self):
        t,a=allan(np.arange(1000)*.03,.1,[1,5,10]);np.testing.assert_allclose(a,.3*t/np.sqrt(2),rtol=1e-10)
    def test_white_expectation(self):
        t,a=allan(np.random.default_rng(12).normal(size=65536),1,[1,4,16]);np.testing.assert_allclose(a,1/np.sqrt(t),rtol=.08)
    def test_invalid_factors(self):
        for f in [[0],[1.5],[51]]:
            with self.assertRaises(ValueError):allan(np.ones(100),1,f)
