import random
import numpy as np
from ..IMetaheuristic import IMetaheuristic
from ..ISolution import SolutionBasic

class BatAlgorithm(IMetaheuristic):
    def __init__(self, to_max=True, population_size=40, fmin=0, fmax=2, 
                 loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9):
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
        self._population_size = population_size
        self._fmin = fmin
        self._fmax = fmax
        self._loudness = loudness
        self._pulse_rate = pulse_rate
        self._alpha = alpha
        self._gamma = gamma
        self._bats = []
        self._velocities = []
        self._frequencies = []
        self._best_solution = None
        
    def initialize_population(self):
        """Initialize the bat population"""
        self._bats = []
        self._velocities = []
        self._frequencies = []
        
        for _ in range(self._population_size):
            # Initialize position
            point = [random.uniform(self.min_x, self.max_x) for _ in range(self.dimension)]
            if self.is_point_valid(point):
                fitness = self.objective_function(point)
                self._bats.append(SolutionBasic(point, fitness))
                
                # Initialize velocity
                velocity = [0.0] * self.dimension
                self._velocities.append(velocity)
                
                # Initialize frequency
                self._frequencies.append(0.0)
        
        self._best_solution = self.find_best_solution(self._bats)
    
    def _local_search(self, bat):
        """Perform local search around the bat"""
        epsilon = random.uniform(-1, 1)
        average_loudness = np.mean([self._loudness])
        new_position = []
        
        for i in range(self.dimension):
            new_pos = bat.point[i] + epsilon * average_loudness
            new_position.append(new_pos)
            
        return self.cut_mod_point(new_position, self.min_x, self.max_x)
    
    def run(self):
        """Execute the Bat Algorithm"""
        self.initialize_population()
        
        for _ in range(self._iterations):
            for i in range(self._population_size):
                # Update frequency
                self._frequencies[i] = self._fmin + (self._fmax - self._fmin) * random.random()
                
                # Update velocity and position
                new_position = []
                for j in range(self.dimension):
                    # Update velocity
                    self._velocities[i][j] = (self._velocities[i][j] + 
                                            (self._bats[i].point[j] - self._best_solution.point[j]) * 
                                            self._frequencies[i])
                    
                    # Update position
                    new_pos = self._bats[i].point[j] + self._velocities[i][j]
                    new_position.append(new_pos)
                
                # Apply bounds
                new_position = self.cut_mod_point(new_position, self.min_x, self.max_x)
                
                # Local search with probability pulse_rate
                if random.random() > self._pulse_rate:
                    new_position = self._local_search(self._bats[i])
                
                # Evaluate new solution
                if self.is_point_valid(new_position):
                    new_fitness = self.objective_function(new_position)
                    
                    # Accept new solution with probability loudness
                    if (random.random() < self._loudness and 
                        ((self._to_max and new_fitness > self._bats[i].fitness) or
                         (not self._to_max and new_fitness < self._bats[i].fitness))):
                        self._bats[i] = SolutionBasic(new_position, new_fitness)
                        
                        # Update pulse rate and loudness
                        self._pulse_rate = self._pulse_rate * (1 - np.exp(-self._gamma))
                        self._loudness *= self._alpha
            
            # Update best solution
            current_best = self.find_best_solution(self._bats)
            if ((self._to_max and current_best.fitness > self._best_solution.fitness) or
                (not self._to_max and current_best.fitness < self._best_solution.fitness)):
                self._best_solution = current_best
        
        return self._best_solution.point, self._best_solution.fitness
