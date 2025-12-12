"""Tests for nonlinear optimization functions."""
import pytest
from anmetal.problems.nonlinear_functions.two_inputs import Camelback, Goldsteinprice

class TestCamelback:
    def test_camelback_properties(self):
        """Test Camelback function properties."""
        limits = Camelback.get_limits()
        assert len(limits) == 2
        assert limits[0] == [-3, 3]
        assert limits[1] == [-2, 2]
        
        assert Camelback.get_type() == "min"
        assert isinstance(Camelback.get_theoretical_optimum(), (int, float))

    def test_camelback_evaluation(self):
        """Test Camelback function evaluation."""
        # Test at optimum (approximate)
        # Global minimum is at (0.0898, -0.7126) and (-0.0898, 0.7126) with value -1.0316
        val = Camelback.func([0.0898, -0.7126])
        assert abs(val - Camelback.get_theoretical_optimum()) < 0.1

class TestGoldsteinprice:
    def test_goldsteinprice_properties(self):
        """Test Goldstein-Price function properties."""
        limits = Goldsteinprice.get_limits()
        assert limits == [-2, 2]
        
        assert Goldsteinprice.get_type() == "min"
        assert Goldsteinprice.get_theoretical_optimum() == 3

    def test_goldsteinprice_evaluation(self):
        """Test Goldstein-Price function evaluation."""
        # Global minimum is at (0, -1) with value 3
        val = Goldsteinprice.func([0, -1])
        assert val == 3
