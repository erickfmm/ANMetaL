"""Tests for optimization problems."""
import pytest
import numpy as np
from anmetal.problems.nphard_categorical.knapsack import Knapsack_Categorical
from anmetal.problems.nphard_categorical.sudoku import Sudoku

class TestKnapsack:
    def test_knapsack_initialization(self):
        """Test knapsack problem initialization."""
        knapsack = Knapsack_Categorical(
            knapsack_capacity=50.8,
            total_posible_elements=50,
            seed=0,
            max_value=5,
            max_cost=6
        )
        assert knapsack.capacity == 50.8
        assert knapsack.total_posible_elements == 50

    def test_knapsack_objective_function(self):
        """Test knapsack objective function calculation."""
        knapsack = Knapsack_Categorical(
            knapsack_capacity=50.8,
            total_posible_elements=5,
            seed=0,
            max_value=5,
            max_cost=6
        )
        solution = [1, 0, 1, 0, 1]  # Example solution
        value = knapsack.objective_function(solution)
        assert isinstance(value, (int, float))
        assert value >= 0

class TestSudoku:
    def test_sudoku_initialization(self):
        """Test sudoku problem initialization."""
        initial_state = [[0] * 9 for _ in range(9)]
        sudoku = Sudoku(initial_state)
        assert len(sudoku.state) == 9
        assert all(len(row) == 9 for row in sudoku.state)

    def test_sudoku_is_valid(self):
        """Test sudoku validity checking."""
        initial_state = [[0] * 9 for _ in range(9)]
        sudoku = Sudoku(initial_state)
        # Test empty board validity
        assert sudoku.is_valid_solution(initial_state)
