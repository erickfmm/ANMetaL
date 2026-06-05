"""Tests for optimization algorithms."""
import pytest
import numpy as np
from anmetal.optimizer.population.AFSA.AFSAMH_Real import AFSAMH_Real
from anmetal.optimizer.population.PSO.PSOMH_Real import PSOMH_Real
from anmetal.problems.nonlinear_functions.two_inputs import Goldsteinprice

class TestAFSA:
    def test_afsa_initialization(self):
        """Test AFSA algorithm initialization."""
        problem = Goldsteinprice()
        afsa = AFSAMH_Real(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        assert afsa._ndims == 2
        assert not afsa._to_max

    def test_afsa_run(self):
        """Test AFSA algorithm execution."""
        problem = Goldsteinprice()
        afsa = AFSAMH_Real(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        fit, pt = afsa.run(
            verbose=False,
            iterations=10,
            population=20,
            visual_distance_percentage=0.5,
            velocity_percentage=0.5,
            seed=115
        )
        assert isinstance(fit, (int, float))
        assert len(pt) == 2

class TestPSO:
    def test_pso_initialization(self):
        """Test PSO algorithm initialization."""
        problem = Goldsteinprice()
        pso = PSOMH_Real(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        assert pso._ndims == 2
        assert not pso._to_max

    def test_pso_run(self):
        """Test PSO algorithm execution."""
        problem = Goldsteinprice()
        pso = PSOMH_Real(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        fit, pt = pso.run(
            verbose=False,
            iterations=10,
            population=20,
            omega=0.8,
            phi_g=0.5,
            phi_p=0.5,
            seed=115
        )
        assert isinstance(fit, (int, float))
        assert len(pt) == 2
