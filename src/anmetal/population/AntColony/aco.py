import random
import numpy as np
from ..IMetaheuristic import IMetaheuristic
from ..ISolution import SolutionBasic
from typing import List, Callable

class AntColony(IMetaheuristic):
    def __init__(self, min_value: float, max_value: float, ndims: int, to_max: bool,
     objective_function: Callable[[List[float]], float],
      repair_function: Callable[[List[float]], List[float]],
      preprocess_function: Callable[[List[float]], List[float]] = None):
        """
        Initialize Ant Colony Optimization
        
        Args:
            to_max: Whether to maximize (True) or minimize (False)
            n_ants: Number of ants in the colony
            evaporation_rate: Rate of pheromone evaporation
            alpha: Pheromone importance factor
            beta: Heuristic information importance factor
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

        self._ants = []
        self._pheromone = None
        self._best_solution = None
        self._discretization_points = 100  # Number of points for discretization
        
    def initialize_population(self, population_size):
        """Initialize the ant colony and pheromone matrix"""
        self._ants = []
        # Initialize ants with random positions
        for _ in range(population_size):
            point = [random.uniform(self._min, self._max) for _ in range(self._ndims)]
            point = self.cut_mod_point(point, self._min, self._max)
            repaired_point = self.repair_function(point)
            fitness = self.objective_function(self.preprocess_function(repaired_point))
            if fitness is not False and fitness is not None:
                self._ants.append(SolutionBasic(repaired_point, fitness))
        
        # Ensure we have at least one valid solution
        while len(self._ants) < 1:
            point = [random.uniform(self._min, self._max) for _ in range(self._ndims)]
            point = self.cut_mod_point(point, self._min, self._max)
            repaired_point = self.repair_function(point)
            fitness = self.objective_function(self.preprocess_function(repaired_point))
            if fitness is not False and fitness is not None:
                self._ants.append(SolutionBasic(repaired_point, fitness))
        
        # Initialize pheromone matrix for each dimension
        # We discretize the continuous space into a grid
        self._pheromone = np.ones((self._ndims, self._discretization_points)) * 0.1
        
        self._best_solution = self.find_best_solution(self._ants)
    
    def _discretize_position(self, position, dim):
        """Convert continuous position to discrete index"""
        normalized = (position - self._min) / (self._max - self._min)
        index = int(normalized * (self._discretization_points - 1))
        return max(0, min(index, self._discretization_points - 1))
    
    def _continuous_position(self, index, dim):
        """Convert discrete index to continuous position"""
        normalized = index / (self._discretization_points - 1)
        return self._min + normalized * (self._max - self._min)
    
    def _construct_solution(self):
        """Construct a new solution for an ant"""
        solution = []
        for dim in range(self._ndims):
            # Calculate probabilities for each discretized point
            pheromone = self._pheromone[dim]
            # Simple heuristic information (can be modified based on problem)
            heuristic = np.ones(self._discretization_points)
            
            probabilities = (pheromone ** self._alpha) * (heuristic ** self._beta)
            probabilities = probabilities / np.sum(probabilities)
            
            # Choose position based on probabilities
            chosen_index = np.random.choice(self._discretization_points, p=probabilities)
            position = self._continuous_position(chosen_index, dim)
            solution.append(position)
        
        return solution
    
    def _update_pheromones(self):
        """Update pheromone levels"""
        # Evaporation
        self._pheromone *= (1 - self._evaporation_rate)
        
        # Add new pheromones based on solutions
        for ant in self._ants:
            deposit = 1.0 / (1.0 + abs(ant.fitness))  # Convert fitness to positive value
            for dim in range(self._ndims):
                idx = self._discretize_position(ant.point[dim], dim)
                self._pheromone[dim][idx] += deposit
        
        # Add extra pheromone for best solution
        best_deposit = 1.0 / (1.0 + abs(self._best_solution.fitness))
        for dim in range(self._ndims):
            idx = self._discretize_position(self._best_solution.point[dim], dim)
            self._pheromone[dim][idx] += best_deposit
    
    def run_yielded(self, iterations: int = 100, population: int = 30, evaporation_rate: float = 0.1, alpha: float = 1.0, beta: float = 2.0, seed: int = None, verbose: bool = False):
        """Execute the Ant Colony Optimization algorithm with yielding"""
        self._iterations = iterations
        self._population = population
        self._seed = seed
        self._n_ants = population
        self._evaporation_rate = evaporation_rate
        self._alpha = alpha
        self._beta = beta
        
        random.seed(seed)
        np.random.seed(seed)

        self.initialize_population(self._n_ants)
        
        iteration = 1
        best_solution_historical = self.find_best_solution(self._ants)
        best_fitness_historical = best_solution_historical.fitness
        best_point_historical = np.copy(best_solution_historical.point)
        
        # yield initial state
        points = [e.point for e in self._ants]
        fts = [e.fitness for e in self._ants]
        bin_point = self.preprocess_function(best_point_historical)
        yield iteration, best_fitness_historical, bin_point, points, fts
        
        while iteration <= iterations:
            if verbose:
                print("it: ", iteration, " fitness mejor: ", best_fitness_historical)
                
            # Generate new solutions for all ants
            new_ants = []
            for _ in range(self._n_ants):
                new_position = self._construct_solution()
                new_position = self.cut_mod_point(new_position, self._min, self._max)
                repaired_position = self.repair_function(new_position)
                fitness = self.objective_function(self.preprocess_function(repaired_position))
                if fitness is not False and fitness is not None:
                    new_ants.append(SolutionBasic(repaired_position, fitness))
            
            if new_ants:  # Only update if we have valid ants
                self._ants = new_ants
            
            # Update best solution
            current_best = self.find_best_solution(self._ants)
            if ((self._to_max and current_best.fitness > best_fitness_historical) or
                (not self._to_max and current_best.fitness < best_fitness_historical)):
                best_fitness_historical = current_best.fitness
                best_point_historical = np.copy(current_best.point)
                self._best_solution = current_best
            
            # Update pheromone trails
            self._update_pheromones()
            
            iteration += 1
                
            # yield current state
            points = [e.point for e in self._ants]
            fts = [e.fitness for e in self._ants]
            bin_point = self.preprocess_function(best_point_historical)
            yield iteration, best_fitness_historical, bin_point, points, fts
        
        # Final yield
        bin_point = self.preprocess_function(best_point_historical)
        points = [e.point for e in self._ants]
        fts = [e.fitness for e in self._ants]
        yield iteration, best_fitness_historical, bin_point, points, fts

    def run(self, iterations: int = 100, population: int = 30, evaporation_rate: float = 0.1, alpha: float = 1.0, beta: float = 2.0, seed: int = None, verbose: bool = False):
        """Execute the Ant Colony Optimization algorithm"""
        for _, best_fitness_historical, bin_point, _, _ in self.run_yielded(
            iterations, population, evaporation_rate, alpha, beta, seed, verbose):
            continue
        return best_fitness_historical, bin_point
