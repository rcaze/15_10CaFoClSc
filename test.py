import unittest
import numpy as np
from lib import tuning_int


class TestLib(unittest.TestCase):
    def test_tuning_int(self):
        # Array of zeros to test tuning_int
        vrec_test = np.array([[[-65, -50],
                               [-65, -65]],
                              [[-65, -65],
                               [-65, -65]]])
        exp_out = np.array([1, 0])
        out = tuning_int(vrec_test, -1, -65)
        print out
        np.testing.assert_array_equal(out, exp_out)
