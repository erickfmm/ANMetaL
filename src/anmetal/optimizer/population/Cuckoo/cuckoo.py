import random
import numpy as np
from ..IMetaheuristic import IMetaheuristic
from ..ISolution import SolutionBasic

class CuckooSearch(IMetaheuristic):
    def __init__(self, min_value, max_value, ndims, to_max=True, objective_function=None, repair_function=None, preprocess_function=None, population_size=25, pa=0.25):
        """
        Initialize Cuckoo Search Algorithm
        
        Args:
            min_value: Minimum value for variables
            max_value: Maximum value for variables
            ndims: Number of dimensions
            to_max: Whether to maximize (True) or minimize (False)
            objective_function: Function to optimize
            repair_function: Function to repair invalid solutions
            preprocess_function: Function to preprocess solutions
            population_size: Number of nests
            pa: Probability of egg abandonment
        """
        super().__init__(to_max)
        self._min = min_value
        self._max = max_value
        self._ndims = ndims
        self._objective_function = objective_function
        self._repair_function = repair_function if repair_function is not None else lambda p: p
        self._preprocess_function = preprocess_function if preprocess_function is not None else lambda p: p
        self._population_size = population_size
        self._pa = pa  # probability of alien eggs discovered
        self._nests = []
        self._best_solution = None
        
    def initialize_population(self):
        """Initialize the population of nests"""
        self._nests = []
        for _ in range(self._population_size):
            point = [random.uniform(self._min, self._max) for _ in range(self._ndims)]
            if self._repair_function:
                point = self._repair_function(point)
            if self._preprocess_function:
                point = self._preprocess_function(point)
                
            fitness = self._objective_function(point)
            if fitness is not False and fitness is not None:
                self._nests.append(SolutionBasic(point, fitness))
        
        # Ensure we have at least one valid solution
        while len(self._nests) < 1:
            point = [random.uniform(self._min, self._max) for _ in range(self._ndims)]
            if self._repair_function:
                point = self._repair_function(point)
            if self._preprocess_function:
                point = self._preprocess_function(point)
                
            fitness = self._objective_function(point)
            if fitness is not False and fitness is not None:
                self._nests.append(SolutionBasic(point, fitness))
        
        if self._nests:
            self._best_solution = self.find_best_solution(self._nests)
    
    def _levy_flight(self):
        """Generate steps using Levy Flight"""
        import math
        beta = 3/2
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
                (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, self._ndims)
        v = np.random.normal(0, 1, self._ndims)
        step = u / abs(v)**(1 / beta)
        return step
    
    def _get_cuckoo(self, current_nest):
        """Generate a new solution via Levy flight"""
        step_size = 0.01  # can be adjusted
        step = self._levy_flight()
        new_position = []
        
        for i in range(self._ndims):
            new_pos = current_nest.point[i] + step_size * step[i]
            new_position.append(new_pos)
            
        # Apply boundary conditions
        new_position = [max(self._min, min(self._max, pos)) for pos in new_position]
        return new_position
    
    def _abandon_nests(self):
        """Abandon worse nests and build new ones"""
        for i in range(len(self._nests)):
            if random.random() < self._pa:
                # Generate new nest
                new_point = [random.uniform(self._min, self._max) for _ in range(self._ndims)]
                if self._repair_function:
                    new_point = self._repair_function(new_point)
                if self._preprocess_function:
                    new_point = self._preprocess_function(new_point)
                    
                fitness = self._objective_function(new_point)
                if fitness is not False and fitness is not None:
                    self._nests[i] = SolutionBasic(new_point, fitness)
    
    def run_yielded(self, iterations: int = 100, population: int = 30, pa: float = 0.25, seed: int = None, verbose: bool = False):
        """Execute the Cuckoo Search algorithm with yielding"""
        self._iterations = iterations
        self._population_size = population  # Update population size
        self._pa = pa  # Update pa parameter
        self._seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.initialize_population()
        
        if not self._nests or not self._best_solution:
            return
            
        iteration = 1
        best_solution_historical = self._best_solution
        best_fitness_historical = best_solution_historical.fitness
        best_point_historical = np.copy(best_solution_historical.point)
        
        # yield initial state
        points = [e.point for e in self._nests]
        fts = [e.fitness for e in self._nests]
        bin_point = self._preprocess_function(best_point_historical)
        yield iteration, best_fitness_historical, bin_point, points, fts
            
        for it in range(iterations):
            if verbose:
                print("it: ", it+1, " fitness mejor: ", best_fitness_historical)
                
            # Get a random nest
            i = random.randint(0, len(self._nests) - 1)
            cuckoo_nest = self._nests[i]
            
            # Generate new solution via Levy flight
            new_position = self._get_cuckoo(cuckoo_nest)
            
            if self._repair_function:
                new_position = self._repair_function(new_position)
            if self._preprocess_function:
                new_position = self._preprocess_function(new_position)
                
            new_fitness = self._objective_function(new_position)
            
            if new_fitness is not False and new_fitness is not None:
                # Random nest to compare with
                j = random.randint(0, len(self._nests) - 1)
                
                # Replace if better
                if ((self._to_max and new_fitness > self._nests[j].fitness) or
                    (not self._to_max and new_fitness < self._nests[j].fitness)):
                    self._nests[j] = SolutionBasic(new_position, new_fitness)
            
            # Abandon worst nests and generate new ones
            self._abandon_nests()
            
            # Update best solution
            if self._nests:
                current_best = self.find_best_solution(self._nests)
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
            points = [e.point for e in self._nests]
            fts = [e.fitness for e in self._nests]
            bin_point = self._preprocess_function(best_point_historical)
            yield iteration, best_fitness_historical, bin_point, points, fts
    
    def run(self, iterations: int = 100, population: int = 30, pa: float = 0.25, seed: int = None, verbose: bool = False):
        """Execute the Cuckoo Search algorithm"""
        for _, best_fitness_historical, bin_point, _, _ in self.run_yielded(
            iterations, population, pa, seed, verbose):
            continue
        return best_fitness_historical, bin_point
