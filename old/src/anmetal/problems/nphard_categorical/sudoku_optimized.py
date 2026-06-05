import numpy as np
from anmetal.problems.IProblem import IProblem

class SudokuOptimized(IProblem):
    def __init__(self, initial_state: list = None):
        self.original_state = np.array(initial_state) if initial_state is not None else np.zeros((9, 9), dtype=int)
        self.state = np.copy(self.original_state)
        
        # Preprocessing: Constraint Propagation
        self.preprocess_constraints()
        
        # Identify fixed cells (non-zero after preprocessing)
        self.fixed_mask = self.state != 0
        
        # Define categories for each cell
        # Fixed cells have only 1 category (their value)
        # Empty cells have categories [1..9]
        self.categories = []
        for r in range(9):
            for c in range(9):
                if self.fixed_mask[r, c]:
                    self.categories.append([int(self.state[r, c])])
                else:
                    self.categories.append([1, 2, 3, 4, 5, 6, 7, 8, 9])

    def preprocess_constraints(self):
        """
        Apply simple constraint propagation (Naked Singles) to reduce search space.
        """
        changed = True
        while changed:
            changed = False
            for r in range(9):
                for c in range(9):
                    if self.state[r, c] == 0:
                        possibles = self.get_possible_values(r, c)
                        if len(possibles) == 1:
                            self.state[r, c] = possibles[0]
                            changed = True

    def get_possible_values(self, r, c):
        """
        Get valid values for cell (r, c) based on current state.
        """
        row_vals = self.state[r, :]
        col_vals = self.state[:, c]
        
        br, bc = (r // 3) * 3, (c // 3) * 3
        box_vals = self.state[br:br+3, bc:bc+3].flatten()
        
        used = set(np.concatenate((row_vals, col_vals, box_vals)))
        used.discard(0) # Remove 0 if present
        
        return [v for v in range(1, 10) if v not in used]

    def get_categories(self):
        return self.categories

    def objective_function(self, point):
        """
        Calculate fitness. 
        Since we enforce Box constraints in repair_function, 
        we only need to count Row and Column violations.
        
        Returns: Number of violations (Minimize this).
        """
        # Point is a list of 81 values. Reshape to 9x9
        grid = np.array(point).reshape((9, 9))
        
        violations = 0
        
        # Row violations: 9 - unique values in row
        for r in range(9):
            violations += (9 - len(np.unique(grid[r, :])))
            
        # Column violations: 9 - unique values in col
        for c in range(9):
            violations += (9 - len(np.unique(grid[:, c])))
            
        return violations

    def preprocess_function(self, point):
        """
        Identity, as we handle preprocessing in init.
        """
        return point

    def repair_function(self, point):
        """
        Enforce 3x3 Box Constraints (Permutation Encoding).
        Ensures each 3x3 box contains numbers 1-9 exactly once.
        Respects fixed values.
        """
        grid = np.array(point).reshape((9, 9))
        
        # Iterate over each 3x3 box
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                box = grid[br:br+3, bc:bc+3]
                box_fixed = self.fixed_mask[br:br+3, bc:bc+3]
                
                # Get values present in the box
                current_vals = box.flatten()
                
                # Identify missing numbers (1-9)
                present_set = set(current_vals)
                missing_nums = [n for n in range(1, 10) if n not in present_set]
                
                # If box is perfect, continue
                if not missing_nums:
                    continue
                
                # Identify mutable positions that have duplicates
                # We need to replace duplicates with missing numbers
                # Strategy: 
                # 1. Keep fixed values.
                # 2. Keep one instance of each valid non-fixed value.
                # 3. Replace duplicates/invalids with missing values.
                
                # Simpler strategy:
                # Collect all fixed values.
                # Collect all mutable positions.
                # Fill mutable positions with remaining numbers (shuffled or ordered).
                
                fixed_vals_in_box = box[box_fixed]
                mutable_indices = []
                for r in range(3):
                    for c in range(3):
                        if not box_fixed[r, c]:
                            mutable_indices.append((r, c))
                
                # Determine which numbers are available for mutable positions
                # Available = {1..9} - {fixed_vals}
                available_nums = set(range(1, 10)) - set(fixed_vals_in_box)
                available_nums = list(available_nums)
                
                # Assign available numbers to mutable positions
                # Note: The input 'point' might have suggested values. 
                # We should try to keep them if they are valid and available.
                # But for strict permutation repair, we can just overwrite.
                # To be "repair" and not "randomize", we should try to keep the gene's intent.
                
                # Better Repair:
                # 1. Identify duplicates in the box (considering fixed ones too).
                # 2. Replace duplicates with missing numbers.
                
                # Let's stick to the "Simpler Strategy" of re-generating the mutable part 
                # to be a permutation of the missing set. 
                # However, to preserve genetic info, we can try to match.
                
                # Implementation of "Replace Duplicates":
                counts = {}
                for x in current_vals:
                    counts[x] = counts.get(x, 0) + 1
                
                missing_iter = iter(missing_nums)
                
                for r in range(3):
                    for c in range(3):
                        if not box_fixed[r, c]:
                            val = box[r, c]
                            if counts[val] > 1:
                                # This is a duplicate (or appeared >1 times)
                                # Replace it with a missing number
                                try:
                                    new_val = next(missing_iter)
                                    box[r, c] = new_val
                                    counts[val] -= 1 # Decrement count of the old value
                                except StopIteration:
                                    pass # Should not happen if math is right
                            # If counts[val] == 1, it's unique, keep it.
                
                grid[br:br+3, bc:bc+3] = box

        return grid.flatten().tolist()
