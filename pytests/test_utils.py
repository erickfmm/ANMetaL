"""Tests for utility functions."""
import pytest
import numpy as np
from anmetal.utils.points_utils import nsphere_to_cartesian, distance, distance_squared, distance_taxicab
from anmetal.utils.binarization_strategy import standard, complement, static_probability, elitist

class TestPointsUtils:
    def test_distance(self):
        """Test Euclidean distance calculation."""
        p1 = [0, 0]
        p2 = [3, 4]
        assert distance(p1, p2) == 5.0
        
        p1 = [1, 1, 1]
        p2 = [1, 1, 1]
        assert distance(p1, p2) == 0.0

    def test_distance_squared(self):
        """Test squared Euclidean distance calculation."""
        p1 = [0, 0]
        p2 = [3, 4]
        assert distance_squared(p1, p2) == 25.0

    def test_distance_taxicab(self):
        """Test Manhattan (Taxicab) distance calculation."""
        p1 = [0, 0]
        p2 = [3, 4]
        assert distance_taxicab(p1, p2) == 7.0

    def test_distance_error(self):
        """Test distance error handling for mismatched dimensions."""
        p1 = [0, 0]
        p2 = [1, 2, 3]
        with pytest.raises(ValueError):
            distance(p1, p2)
        with pytest.raises(ValueError):
            distance_squared(p1, p2)
        with pytest.raises(ValueError):
            distance_taxicab(p1, p2)

class TestBinarizationStrategy:
    def test_standard(self):
        """Test standard binarization."""
        assert standard(0.6, 0.5) == 1
        assert standard(0.4, 0.5) == 0

    def test_elitist(self):
        """Test elitist binarization."""
        # elitist(x, u1, u2): return standard(x, u2) if u1 < x else 0
        # if u1 < x: check standard(x, u2) -> if u2 <= x return 1 else 0
        # if u1 >= x: return 0
        
        # Case 1: u1 < x, u2 <= x -> 1
        assert elitist(0.7, 0.5, 0.6) == 1
        # Case 2: u1 < x, u2 > x -> 0
        assert elitist(0.7, 0.5, 0.8) == 0
        # Case 3: u1 >= x -> 0
        assert elitist(0.4, 0.5, 0.1) == 0
