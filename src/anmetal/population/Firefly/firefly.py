import random
import numpy as np
from ..IMetaheuristic import IMetaheuristic
from ..ISolution import SolutionBasic

class FireflyAlgorithm(IMetaheuristic):
    def __init__(self, to_max=True, population_size=40, alpha=0.5, beta0=1.0, gamma=1.0):
        """
        Initialize Firefly Algorithm
        
        Args:
            to_max: Whether to maximize (True) or minimize (False)
            population_size: Number of fireflies
            alpha: Randomization parameter
            beta0: Attractiveness at distance=0
            gamma: Light absorption coefficient
        """
        super().__init__(to_max)
        self._population_size = population_size
        self._alpha = alpha
        self._beta0 = beta0
        self._gamma = gamma
        self._fireflies = []
        self._best_solution = None
    
    def initialize_population(self):
        """Initialize the population of fireflies"""
        self._fireflies = []
        for _ in range(self._population_size):
            point = [random.uniform(self.min_x, self.max_x) for _ in range(self.dimension)]
            if self.is_point_valid(point):
                fitness = self.objective_function(point)
                self._fireflies.append(SolutionBasic(point, fitness))
        
        self._best_solution = self.find_best_solution(self._fireflies)
    
    def _distance(self, firefly1, firefly2):
        """Calculate the Cartesian distance between two fireflies"""
        return np.sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(firefly1.point, firefly2.point)))
    
    def _attract_and_move(self, firefly1, firefly2):
        """Move firefly1 towards firefly2 if firefly2 is brighter"""
        distance = self._distance(firefly1, firefly2)
        beta = self._beta0 * np.exp(-self._gamma * distance ** 2)
        
        new_position = []
        for i in range(self.dimension):
            # Movement towards brighter firefly + random movement
            rand = random.uniform(-0.5, 0.5)
            movement = (beta * (firefly2.point[i] - firefly1.point[i]) + 
                       self._alpha * rand)
            new_pos = firefly1.point[i] + movement
            new_position.append(new_pos)
        
        return self.cut_mod_point(new_position, self.min_x, self.max_x)
    
    def run(self):
        """Execute the Firefly Algorithm"""
        self.initialize_population()
        
        for _ in range(self._iterations):
            # For each firefly
            for i in range(self._population_size):
                # Compare with all other fireflies
                for j in range(self._population_size):
                    if i == j:
                        continue
                        
                    # Move if the other firefly is brighter
                    if ((self._to_max and self._fireflies[j].fitness > self._fireflies[i].fitness) or
                        (not self._to_max and self._fireflies[j].fitness < self._fireflies[i].fitness)):
                        new_position = self._attract_and_move(self._fireflies[i], self._fireflies[j])
                        
                        if self.is_point_valid(new_position):
                            new_fitness = self.objective_function(new_position)
                            self._fireflies[i] = SolutionBasic(new_position, new_fitness)
            
            # Update best solution
            current_best = self.find_best_solution(self._fireflies)
            if ((self._to_max and current_best.fitness > self._best_solution.fitness) or
                (not self._to_max and current_best.fitness < self._best_solution.fitness)):
                self._best_solution = current_best
            
            # Reduce alpha (optional)
            self._alpha *= 0.97
        
        return self._best_solution.point, self._best_solution.fitness
