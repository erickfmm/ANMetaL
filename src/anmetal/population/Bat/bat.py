import random
import numpy as np
from ..IMetaheuristic import IMetaheuristic
from ..ISolution import SolutionBasic
from typing import List, Callable

class BatAlgorithm(IMetaheuristic):
    def __init__(self, min_value: float, max_value: float, ndims: int, to_max: bool,
     objective_function: Callable[[List[float]], float],
      repair_function: Callable[[List[float]], List[float]],
      preprocess_function: Callable[[List[float]], List[float]] = None):
        """
        Initialize Bat Algorithm
        
        Args:
            to_max: Whether to maximize (True) or minimize (False)
            population_size: Number of bats
            fmin: Minimum frequency
            fmax: Maximum frequency
            loudness: Initial loudness
            pulse_rate: Initial pulse rate
            alpha: Loudness reduction constant
            gamma: Pulse rate increase constant
        """
        super().__init__(to_max)
        self._min = min_value
        self._max = max_value
        self._ndims = ndims
        self._to_max = to_max

        self.objective_function = objective_function
        self.preprocess_function = \
         preprocess_function if preprocess_function is not None else lambda p: p
        self.repair_function = \
         repair_function if repair_function is not None else lambda p: p

        self._bats = []
        self._velocities = []
        self._frequencies = []
        self._best_solution = None
        
    def initialize_population(self, population_size):
        """Initialize the bat population"""
        self._bats = []
        self._velocities = []
        self._frequencies = []
        
        for _ in range(population_size):
            # Initialize position
            point = [random.uniform(self._min, self._max) for _ in range(self._ndims)]
            point = self.cut_mod_point(point, self._min, self._max)
            repaired_point = self.repair_function(point)
            fitness = self.objective_function(self.preprocess_function(repaired_point))
            if fitness is not False and fitness is not None:
                self._bats.append(SolutionBasic(repaired_point, fitness))
                
                # Initialize velocity
                velocity = [0.0] * self._ndims
                self._velocities.append(velocity)
                
                # Initialize frequency
                self._frequencies.append(0.0)
        
        # Ensure we have at least one valid solution
        while len(self._bats) < 1:
            point = [random.uniform(self._min, self._max) for _ in range(self._ndims)]
            point = self.cut_mod_point(point, self._min, self._max)
            repaired_point = self.repair_function(point)
            fitness = self.objective_function(self.preprocess_function(repaired_point))
            if fitness is not False and fitness is not None:
                self._bats.append(SolutionBasic(repaired_point, fitness))
                velocity = [0.0] * self._ndims
                self._velocities.append(velocity)
                self._frequencies.append(0.0)
        
        self._best_solution = self.find_best_solution(self._bats)
    
    def _local_search(self, bat):
        """Perform local search around the bat"""
        epsilon = random.uniform(-1, 1)
        average_loudness = self._loudness
        new_position = []
        
        for i in range(self._ndims):
            new_pos = bat.point[i] + epsilon * average_loudness
            new_position.append(new_pos)
            
        return self.cut_mod_point(new_position, self._min, self._max)
    
    def run_yielded(self, iterations: int = 100, population: int = 30, fmin: float = 0, fmax: float = 2, A: float = 0.9, r0: float = 0.9, seed: int = None, verbose: bool = False):
        """Execute the Bat Algorithm with yielding"""
        self._iterations = iterations
        self._population = population
        self._seed = seed
        self._population_size = population
        self._fmin = fmin
        self._fmax = fmax
        self._loudness = A
        self._pulse_rate = r0
        self._alpha = 0.9
        self._gamma = 0.9
        
        random.seed(seed)
        np.random.seed(seed)

        self.initialize_population(self._population_size)
        
        iteration = 1
        best_solution_historical = self.find_best_solution(self._bats)
        best_fitness_historical = best_solution_historical.fitness
        best_point_historical = np.copy(best_solution_historical.point)
        
        # yield initial state
        points = [e.point for e in self._bats]
        fts = [e.fitness for e in self._bats]
        bin_point = self.preprocess_function(best_point_historical)
        yield iteration, best_fitness_historical, bin_point, points, fts
        
        while iteration <= iterations:
            if verbose:
                print("it: ", iteration, " fitness mejor: ", best_fitness_historical)
                
            for i in range(self._population_size):
                # Update frequency
                self._frequencies[i] = self._fmin + (self._fmax - self._fmin) * random.random()
                
                # Update velocity and position
                new_position = []
                for j in range(self._ndims):
                    # Update velocity
                    self._velocities[i][j] = (self._velocities[i][j] + 
                                            (self._bats[i].point[j] - self._best_solution.point[j]) * 
                                            self._frequencies[i])
                    
                    # Update position
                    new_pos = self._bats[i].point[j] + self._velocities[i][j]
                    new_position.append(new_pos)
                
                # Apply bounds
                new_position = self.cut_mod_point(new_position, self._min, self._max)
                
                # Local search with probability pulse_rate
                if random.random() > self._pulse_rate:
                    new_position = self._local_search(self._bats[i])
                
                # Evaluate new solution
                repaired_position = self.repair_function(new_position)
                new_fitness = self.objective_function(self.preprocess_function(repaired_position))
                
                # Accept new solution with probability loudness
                if (new_fitness is not False and new_fitness is not None and
                    random.random() < self._loudness and 
                    ((self._to_max and new_fitness > self._bats[i].fitness) or
                     (not self._to_max and new_fitness < self._bats[i].fitness))):
                    self._bats[i] = SolutionBasic(repaired_position, new_fitness)
                    
                    # Update pulse rate and loudness
                    self._pulse_rate = self._pulse_rate * (1 - np.exp(-self._gamma))
                    self._loudness *= self._alpha
            
            # Update best solution
            current_best = self.find_best_solution(self._bats)
            if ((self._to_max and current_best.fitness > best_fitness_historical) or
                (not self._to_max and current_best.fitness < best_fitness_historical)):
                best_fitness_historical = current_best.fitness
                best_point_historical = np.copy(current_best.point)
                self._best_solution = current_best
            
            iteration += 1
                
            # yield current state
            points = [e.point for e in self._bats]
            fts = [e.fitness for e in self._bats]
            bin_point = self.preprocess_function(best_point_historical)
            yield iteration, best_fitness_historical, bin_point, points, fts
        
        # Final yield
        bin_point = self.preprocess_function(best_point_historical)
        points = [e.point for e in self._bats]
        fts = [e.fitness for e in self._bats]
        yield iteration, best_fitness_historical, bin_point, points, fts

    def run(self, iterations: int = 100, population: int = 30, fmin: float = 0, fmax: float = 2, A: float = 0.9, r0: float = 0.9, seed: int = None, verbose: bool = False):
        """Execute the Bat Algorithm"""
        for _, best_fitness_historical, bin_point, _, _ in self.run_yielded(
            iterations, population, fmin, fmax, A, r0, seed, verbose):
            continue
        return best_fitness_historical, bin_point
