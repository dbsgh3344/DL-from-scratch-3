import unittest
import sys
import os
sys.path.append(os.path.join(os.getcwd(),"study"))
from base_class import Variable,Function,np
from functions import square,exp

def centered_diff(f:Function,x:Variable) :
    h = 1e-4
    x0 = Variable(x.data + h)
    x1 = Variable(x.data - h)
    y0 = f(x0)
    y1 = f(x1)
    return (y0.data - y1.data) / (2*h)


class SquareTest(unittest.TestCase) :
    def test_forward(self) :
        x = Variable(np.array(2.0))
        result = square(x)
        expected_result = np.array(4.0)
        self.assertEqual(result.data,expected_result)

    def test_backward(self) :
        x = Variable(np.array(2.0))
        y = square(x)        
        y.backward()
        expected_result = np.array(4.0)
        self.assertEqual(x.grad,expected_result)

    def test_grad_check(self) :
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        expected_result = centered_diff(square,x)
        flg = np.allclose(x.grad,expected_result)
        self.assertTrue(flg)

