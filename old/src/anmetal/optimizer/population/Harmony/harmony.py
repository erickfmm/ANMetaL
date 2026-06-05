import random
import numpy as np
from ..IMetaheuristic import IMetaheuristic
from ..ISolution import SolutionBasic

class HarmonySearch(IMetaheuristic):
    def __init__(self, min_value, max_value, ndims, to_max, objective_function, repair_function, preprocess_function=None):
        """
        Initialize Harmony Search Algorithm
        
        Args:
            min_value: Minimum value for variables
            max_value: Maximum value for variables
            ndims: Number of dimensions
            to_max: Whether to maximize (True) or minimize (False)
            objective_function: Function to optimize
            repair_function: Function to repair invalid solutions
            preprocess_function: Function to preprocess solutions
        """
        super().__init__(to_max)
        self._min = min_value
        self._max = max_value
        self._ndims = ndims
        self._to_max = to_max
        self._objective_function = objective_function
        self._repair_function = repair_function if repair_function is not None else lambda p: p
        self._preprocess_function = preprocess_function if preprocess_function is not None else lambda p: p
        # Default parameters - will be overridden in run methods
        self._hms = 30  # harmony memory size
        self._hmcr = 0.9  # harmony memory considering rate
        self._par = 0.3  # pitch adjustment rate
        self._bw = 0.01  # bandwidth
        self._harmony_memory = []
        self._best_solution = None
    
    def initialize_population(self):
        """Initialize the harmony memory"""
        self._harmony_memory = []
        for _ in range(self._hms):
            point = [random.uniform(self._min, self._max) for _ in range(self._ndims)]
            if self._repair_function:
                point = self._repair_function(point)
                
            fitness = self._objective_function(self._preprocess_function(point))
            if fitness is not False and fitness is not None:
                self._harmony_memory.append(SolutionBasic(point, fitness))
        
        if self._harmony_memory:
            self._best_solution = self.find_best_solution(self._harmony_memory)
    
    def _memory_consideration(self, dimension):
        """Select a value from harmony memory"""
        random_harmony = random.choice(self._harmony_memory)
        return random_harmony.point[dimension]
    
    def _pitch_adjustment(self, value):
        """Adjust the pitch of a note"""
        if random.random() < self._par:
            return value + self._bw * random.uniform(-1, 1)
        return value
    
    def _create_new_harmony(self):
        """Generate a new harmony"""
        new_harmony = []
        for i in range(self._ndims):
            if random.random() < self._hmcr:
                # Memory consideration
                value = self._memory_consideration(i)
                # Pitch adjustment
                value = self._pitch_adjustment(value)
            else:
                # Random selection
                value = random.uniform(self._min, self._max)
            new_harmony.append(value)
        
        # Apply boundary conditions
        return [max(self._min, min(self._max, val)) for val in new_harmony]
    
    def _update_harmony_memory(self, new_harmony, new_fitness):
        """Update harmony memory if new harmony is better than worst harmony"""
        worst_index = 0
        worst_fitness = self._harmony_memory[0].fitness
        
        for i in range(1, self._hms):
            if ((self._to_max and self._harmony_memory[i].fitness < worst_fitness) or
                (not self._to_max and self._harmony_memory[i].fitness > worst_fitness)):
                worst_index = i
                worst_fitness = self._harmony_memory[i].fitness
        
        if ((self._to_max and new_fitness > worst_fitness) or
            (not self._to_max and new_fitness < worst_fitness)):
            self._harmony_memory[worst_index] = SolutionBasic(new_harmony, new_fitness)
    
    def run_yielded(self, iterations: int = 100, population: int = 30, hmcr: float = 0.9, par: float = 0.3, bw: float = 0.2, seed: int = None, verbose: bool = False):
        """Execute the Harmony Search algorithm with yielding"""
        self._iterations = iterations
        self._hms = population  # Use population as harmony memory size
        self._hmcr = hmcr  # Update harmony memory considering rate
        self._par = par  # Update pitch adjustment rate
        self._bw = bw  # Update bandwidth
        self._seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.initialize_population()
        
        if not self._harmony_memory:
            return
            
        self._best_solution = self.find_best_solution(self._harmony_memory)
        if not self._best_solution:
            return
            
        iteration = 1
        best_solution_historical = self._best_solution
        best_fitness_historical = best_solution_historical.fitness
        best_point_historical = np.copy(best_solution_historical.point)
        
        # yield initial state
        points = [e.point for e in self._harmony_memory]
        fts = [e.fitness for e in self._harmony_memory]
        bin_point = self._preprocess_function(best_point_historical)
        yield iteration, best_fitness_historical, bin_point, points, fts
            
        for it in range(iterations):
            if verbose:
                print("it: ", it+1, " fitness mejor: ", best_fitness_historical)
                
            # Create new harmony
            new_harmony = self._create_new_harmony()
            
            if self._repair_function:
                new_harmony = self._repair_function(new_harmony)
                
            # Evaluate new harmony
            new_fitness = self._objective_function(self._preprocess_function(new_harmony))
            
            if new_fitness is not False and new_fitness is not None:
                # Update harmony memory
                self._update_harmony_memory(new_harmony, new_fitness)
                
                # Update best solution
                if self._harmony_memory:
                    current_best = self.find_best_solution(self._harmony_memory)
                    if current_best and self._best_solution:
                        if ((self._to_max and current_best.fitness > self._best_solution.fitness) or
                            (not self._to_max and current_best.fitness < self._best_solution.fitness)):
                            self._best_solution = current_best
            
            iteration += 1
            
            # Update best solution
            if ((self._to_max and self._best_solution.fitness > best_fitness_historical) or
                (not self._to_max and self._best_solution.fitness < best_fitness_historical)):
                best_fitness_historical = self._best_solution.fitness
                best_point_historical = np.copy(self._best_solution.point)
                
            # yield current state
            points = [e.point for e in self._harmony_memory]
            fts = [e.fitness for e in self._harmony_memory]
            bin_point = self._preprocess_function(best_point_historical)
            yield iteration, best_fitness_historical, bin_point, points, fts
    
    def run(self, iterations: int = 100, population: int = 30, hmcr: float = 0.9, par: float = 0.3, bw: float = 0.2, seed: int = None, verbose: bool = False):
        """Execute the Harmony Search algorithm"""
        for _, best_fitness_historical, bin_point, _, _ in self.run_yielded(
            iterations, population, hmcr, par, bw, seed, verbose):
            continue
        return best_fitness_historical, bin_point
