import random
import numpy as np
from ..IMetaheuristic import IMetaheuristic
from ..ISolution import SolutionBasic

class AntColony(IMetaheuristic):
    def __init__(self, to_max=True, n_ants=30, evaporation_rate=0.1, alpha=1.0, beta=2.0):
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
        self._n_ants = n_ants
        self._evaporation_rate = evaporation_rate
        self._alpha = alpha
        self._beta = beta
        self._ants = []
        self._pheromone = None
        self._best_solution = None
        self._discretization_points = 100  # Number of points for discretization
        
    def initialize_population(self):
        """Initialize the ant colony and pheromone matrix"""
        self._ants = []
        # Initialize ants with random positions
        for _ in range(self._n_ants):
            point = [random.uniform(self.min_x, self.max_x) for _ in range(self.dimension)]
            if self.is_point_valid(point):
                fitness = self.objective_function(point)
                self._ants.append(SolutionBasic(point, fitness))
        
        # Initialize pheromone matrix for each dimension
        # We discretize the continuous space into a grid
        self._pheromone = np.ones((self.dimension, self._discretization_points)) * 0.1
        
        self._best_solution = self.find_best_solution(self._ants)
    
    def _discretize_position(self, position, dim):
        """Convert continuous position to discrete index"""
        normalized = (position - self.min_x) / (self.max_x - self.min_x)
        index = int(normalized * (self._discretization_points - 1))
        return max(0, min(index, self._discretization_points - 1))
    
    def _continuous_position(self, index, dim):
        """Convert discrete index to continuous position"""
        normalized = index / (self._discretization_points - 1)
        return self.min_x + normalized * (self.max_x - self.min_x)
    
    def _construct_solution(self):
        """Construct a new solution for an ant"""
        solution = []
        for dim in range(self.dimension):
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
            for dim in range(self.dimension):
                idx = self._discretize_position(ant.point[dim], dim)
                self._pheromone[dim][idx] += deposit
        
        # Add extra pheromone for best solution
        best_deposit = 1.0 / (1.0 + abs(self._best_solution.fitness))
        for dim in range(self.dimension):
            idx = self._discretize_position(self._best_solution.point[dim], dim)
            self._pheromone[dim][idx] += best_deposit
    
    def run(self):
        """Execute the Ant Colony Optimization algorithm"""
        self.initialize_population()
        
        for _ in range(self._iterations):
            # Generate new solutions for all ants
            new_ants = []
            for _ in range(self._n_ants):
                new_position = self._construct_solution()
                if self.is_point_valid(new_position):
                    fitness = self.objective_function(new_position)
                    new_ants.append(SolutionBasic(new_position, fitness))
            
            self._ants = new_ants
            
            # Update best solution
            current_best = self.find_best_solution(self._ants)
            if ((self._to_max and current_best.fitness > self._best_solution.fitness) or
                (not self._to_max and current_best.fitness < self._best_solution.fitness)):
                self._best_solution = current_best
            
            # Update pheromone trails
            self._update_pheromones()
        
        return self._best_solution.point, self._best_solution.fitness
