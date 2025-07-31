import random
import numpy as np
from ..IMetaheuristic import IMetaheuristic
from ..ISolution import SolutionBasic

class FireflyAlgorithm(IMetaheuristic):
    def __init__(self, min_value, max_value, ndims, to_max, objective_function, repair_function, preprocess_function=None):
        """
        Initialize Firefly Algorithm
        
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
        self._population_size = 30
        self._alpha = 0.5
        self._beta0 = 1.0
        self._gamma = 1.0
        self._fireflies = []
        self._best_solution = None
    
    def initialize_population(self):
        """Initialize the population of fireflies"""
        self._fireflies = []
        for _ in range(self._population_size):
            point = [random.uniform(self._min, self._max) for _ in range(self._ndims)]
            if self._repair_function:
                point = self._repair_function(point)
                
            fitness = self._objective_function(self._preprocess_function(point))
            if fitness is not False and fitness is not None:
                self._fireflies.append(SolutionBasic(point, fitness))
        
        if self._fireflies:
            self._best_solution = self.find_best_solution(self._fireflies)
    
    def _distance(self, firefly1, firefly2):
        """Calculate the Cartesian distance between two fireflies"""
        return np.sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(firefly1.point, firefly2.point)))
    
    def _attract_and_move(self, firefly1, firefly2):
        """Move firefly1 towards firefly2 if firefly2 is brighter"""
        distance = self._distance(firefly1, firefly2)
        beta = self._beta0 * np.exp(-self._gamma * distance ** 2)
        
        new_position = []
        for i in range(self._ndims):
            # Movement towards brighter firefly + random movement
            rand = random.uniform(-0.5, 0.5)
            movement = (beta * (firefly2.point[i] - firefly1.point[i]) + 
                       self._alpha * rand)
            new_pos = firefly1.point[i] + movement
            new_position.append(new_pos)
        
        # Apply boundary conditions
        return [max(self._min, min(self._max, pos)) for pos in new_position]
    
    def run_yielded(self, iterations: int = 100, population: int = 30, alpha: float = 0.5, beta0: float = 1.0, gamma: float = 1.0, seed: int = None, verbose: bool = False):
        """Execute the Firefly Algorithm with yielding"""
        self._iterations = iterations
        self._population_size = population  # Update population size
        self._alpha = alpha  # Update alpha parameter
        self._beta0 = beta0  # Update beta0 parameter
        self._gamma = gamma  # Update gamma parameter
        self._seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.initialize_population()
        
        if not self._fireflies or not self._best_solution:
            return
            
        iteration = 1
        best_solution_historical = self._best_solution
        best_fitness_historical = best_solution_historical.fitness
        best_point_historical = np.copy(best_solution_historical.point)
        
        # yield initial state
        points = [e.point for e in self._fireflies]
        fts = [e.fitness for e in self._fireflies]
        bin_point = self._preprocess_function(best_point_historical)
        yield iteration, best_fitness_historical, bin_point, points, fts
            
        for it in range(iterations):
            if verbose:
                print("it: ", it+1, " fitness mejor: ", best_fitness_historical)
                
            # For each firefly
            for i in range(len(self._fireflies)):
                # Compare with all other fireflies
                for j in range(len(self._fireflies)):
                    if i == j:
                        continue
                        
                    # Move if the other firefly is brighter
                    if ((self._to_max and self._fireflies[j].fitness > self._fireflies[i].fitness) or
                        (not self._to_max and self._fireflies[j].fitness < self._fireflies[i].fitness)):
                        new_position = self._attract_and_move(self._fireflies[i], self._fireflies[j])
                        
                        if self._repair_function:
                            new_position = self._repair_function(new_position)
                            
                        new_fitness = self._objective_function(self._preprocess_function(new_position))
                        if new_fitness is not False and new_fitness is not None:
                            self._fireflies[i] = SolutionBasic(new_position, new_fitness)
            
            # Update best solution
            if self._fireflies:
                current_best = self.find_best_solution(self._fireflies)
                if current_best and self._best_solution:
                    if ((self._to_max and current_best.fitness > self._best_solution.fitness) or
                        (not self._to_max and current_best.fitness < self._best_solution.fitness)):
                        self._best_solution = current_best
            
            # Reduce alpha (optional)
            self._alpha *= 0.97
            
            iteration += 1
            
            # Update best solution
            if ((self._to_max and self._best_solution.fitness > best_fitness_historical) or
                (not self._to_max and self._best_solution.fitness < best_fitness_historical)):
                best_fitness_historical = self._best_solution.fitness
                best_point_historical = np.copy(self._best_solution.point)
                
            # yield current state
            points = [e.point for e in self._fireflies]
            fts = [e.fitness for e in self._fireflies]
            bin_point = self._preprocess_function(best_point_historical)
            yield iteration, best_fitness_historical, bin_point, points, fts
    
    def run(self, iterations: int = 100, population: int = 30, alpha: float = 0.5, beta0: float = 1.0, gamma: float = 1.0, seed: int = None, verbose: bool = False):
        """Execute the Firefly Algorithm"""
        for _, best_fitness_historical, bin_point, _, _ in self.run_yielded(
            iterations, population, alpha, beta0, gamma, seed, verbose):
            continue
        return best_fitness_historical, bin_point
