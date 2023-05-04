import unittest
import numpy as np
from nn.activation_functions import relu, linear, softmax

class TestActivationFunctions(unittest.TestCase):
    def test_relu(self):

        # simple
        x = np.array([-1,-2,-3, 0,1,2,3,4])
        y = np.array([0,0,0,0,1,2,3,4])

        np.testing.assert_equal(relu(x), y)

        # different shape
        np.testing.assert_equal(relu(x.reshape((2,2,2))), y.reshape((2,2,2)))
    
    def test_softmax(self):

        # simple
        x = np.array([-1,-2,-3, 0,1,2,3,4])
        y = np.array([4.260624e-03, 1.567396e-03, 5.766128e-04, 1.158158e-02,
       3.148199e-02, 8.557692e-02, 2.326222e-01, 6.323327e-01])
        np.testing.assert_array_almost_equal(softmax(x), y)

        # different shape
        np.testing.assert_array_almost_equal(softmax(x.reshape((2,2,2))), y.reshape((2,2,2)))

    def test_linear(self):

        # simple
        x = np.array([-1,-2,-3, 0,1,2,3,4])
        y = np.array([-1,-2,-3, 0,1,2,3,4])

        np.testing.assert_equal(linear(x), y)

        # different shape
        np.testing.assert_equal(linear(x.reshape((2,2,2))), y.reshape((2,2,2)))

if __name__ == "__main__":
    unittest.main()