"""Tests for single solution iterative methods."""
import pytest
import math
from anmetal.optimizer.single_solution.iterative_methods import euler_method, newton_method

class TestIterativeMethods:
    def test_euler_method(self):
        """Test Euler method for solving ODEs."""
        # dy/dx = y, y(0) = 1 -> y(x) = e^x
        # Estimate y(1)
        def dy_dx(x, y):
            return y
            
        y_est = euler_method(dy_dx, 0, 1, 1, 1000)
        y_true = math.exp(1)
        
        # Euler method is first order, so error should be roughly proportional to step size (1/1000)
        assert abs(y_est - y_true) < 0.01

    def test_newton_method(self):
        """Test Newton method for finding roots."""
        # f(x) = x^2 - 4, root at 2
        def func(x):
            return x**2 - 4
            
        def deriv(x):
            return 2*x
            
        root = newton_method(func, deriv, 3, 10)
        assert abs(root - 2.0) < 1e-6
        
        # Test with multiplicity m=1 (default)
        root = newton_method(func, deriv, 3, 10, m=1)
        assert abs(root - 2.0) < 1e-6
