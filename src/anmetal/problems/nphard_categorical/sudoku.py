#This file implements the sudoku solving problem
import numpy as np
from anmetal.problems.IProblem import IProblem
class Sudoku(IProblem):
    def __init__(self, initial_state: list=None):
        self.state = initial_state
        if self.state is None:
            self.state = np.zeros((9,9))
        self.categories = [1,2,3,4,5,6,7,8,9]

    def set_categories(self, categories):
        self.categories = categories

    def is_row_violating(self, row, comming_from: str = ""):
        if len(row) != 9:
            raise ValueError("state row, column or box should be of size 9. comming from: "+comming_from)
        ncats = 0
        violated_row = False
        for x in row:
            if x != 0 and x not in self.categories:  # Allow 0 for empty cells
                violated_row = True
                break
            if x != 0:  # Only count non-empty cells
                equals_filter = row == x
                if np.sum(equals_filter) > 1:
                    violated_row = True
                    break
                ncats += 1
        return violated_row

    def get_violations(self):
        if not isinstance(self.state, np.ndarray):
            raise ValueError("state should be numpy ndarray")
        n_violations = 0
        #horizontal
        for ihoriz in range(9):
            if self.is_row_violating(self.state[ihoriz], "horizontal: "+str(ihoriz)):
                n_violations += 1
        #vertical
        for ivert in range(9):
            if self.is_row_violating(self.state[:,ivert], "vertical: "+str(ivert)):
                n_violations += 1
        #boxes
        for ih1 in [0,3,6]:
            ih2 = ih1+3
            for ivert1 in [0,3,6]:
                ivert2 = ivert1 + 3
                box = self.state[ih1:ih2,ivert1:ivert2]
                if self.is_row_violating(box.flatten(), "box ["+str(ih1)+":"+str(ih2)+","+str(ivert1)+":"+str(ivert2)+"]"):
                    n_violations +=1
        return n_violations

    def is_valid_solution(self, solution=None):
        """Check if the current state or provided solution is valid."""
        if solution is not None:
            old_state = self.state
            self.state = np.array(solution)
            is_valid = self.get_violations() == 0
            self.state = old_state
            return is_valid
        return self.get_violations() == 0
