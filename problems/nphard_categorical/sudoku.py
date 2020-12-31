import numpy as np

class Sudoku:
    def __init__(self, initial_state: list=None):
        self.state = initial_state
        if self.state is None:
            self.state = np.zeros((9,9))
        self.categories = [1,2,3,4,5,6,7,8,9]

    def set_categories(self, categories):
        self.categories = categories

    @staticmethod
    def is_row_violating(row, comming_from: str = ""):
        if len(row) != 9:
            ValueError("state row, column or box should be of size 9. comming from: "+comming_from)
        ncats = 0
        violated_row = False
        for x in row:
            if x not in self.categories:
                violated_row = True
                break
            equals_filter = row == x
            if np.sum(equals_filter) > 1:
                violated_row = True
                break
            ncats +=1
        if violated_row or ncats != 9:
            return True
        return False

    def get_violations(self):
        if type(self.state) != type(np.zeros(1)):
            raise ValueError("state should be numpy ndarray")
        n_violations = 0
        #horizontal
        for ihoriz in range(1,10):
            if Sudoku.is_row_violating(self.state[ihoriz], "horizontal: "+str(ihoriz)):
                n_violations += 1
        #vertical
        for ivert in range(1,10):
            if Sudoku.is_row_violating(self.state[:,ivert], "vertical: "+str(ivert)):
                n_violations += 1
        #boxes
        #self.states
        #[0:3,0:3]  [0:3,3:6]   [0:3,6:9]
        #[3:6,0:3]  [3:6,0:3]   [3:6,6:9]
        #[6:9,0:3]  [6:9,0:3]   [6:9,6:9]
        for ih1 in [0,3,6]:
            ih2 = ih1+3
            for ivert1 in [0,3,6]:
                ivert2 = ivert1 + 3
                box = self.state[ih1:ih2,ivert1:ivert2]
                if Sudoku.is_row_violating(box.flatten(), "box ["+str(ih1)+":"+str(ih2)+","+str(ivert1)+":"+str(ivert2)+"]"):
                    n_violations +=1
