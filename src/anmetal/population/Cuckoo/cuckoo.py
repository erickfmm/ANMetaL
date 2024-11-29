import random
import numpy as np
from ..IMetaheuristic import IMetaheuristic
from ..ISolution import SolutionBasic

class CuckooSearch(IMetaheuristic):
    def __init__(self, to_max=True, population_size=25, pa=0.25):
        """
        Initialize Cuckoo Search Algorithm
        
        Args:
            to_max: Whether to maximize (True) or minimize (False)
            population_size: Number of nests
            pa: Probability of egg abandonment
        """
        super().__init__(to_max)
        self._population_size = population_size
        self._pa = pa  # probability of alien eggs discovered
        self._nests = []
        self._best_solution = None
        
    def initialize_population(self):
        """Initialize the population of nests"""
        self._nests = []
        for _ in range(self._population_size):
            point = [random.uniform(self.min_x, self.max_x) for _ in range(self.dimension)]
            if self.is_point_valid(point):
                fitness = self.objective_function(point)
                self._nests.append(SolutionBasic(point, fitness))
        
        self._best_solution = self.find_best_solution(self._nests)
    
    def _levy_flight(self):
        """Generate steps using Levy Flight"""
        beta = 3/2
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, self.dimension)
        v = np.random.normal(0, 1, self.dimension)
        step = u / abs(v)**(1 / beta)
        return step
    
    def _get_cuckoo(self, current_nest):
        """Generate a new solution via Levy flight"""
        step_size = 0.01  # can be adjusted
        step = self._levy_flight()
        new_position = []
        
        for i in range(self.dimension):
            new_pos = current_nest.point[i] + step_size * step[i]
            new_position.append(new_pos)
            
        new_position = self.cut_mod_point(new_position, self.min_x, self.max_x)
        return new_position
    
    def _abandon_nests(self):
        """Abandon worse nests and build new ones"""
        for i in range(len(self._nests)):
            if random.random() < self._pa:
                # Generate new nest
                new_point = [random.uniform(self.min_x, self.max_x) for _ in range(self.dimension)]
                if self.is_point_valid(new_point):
                    fitness = self.objective_function(new_point)
                    self._nests[i] = SolutionBasic(new_point, fitness)
    
    def run(self):
        """Execute the Cuckoo Search algorithm"""
        self.initialize_population()
        
        for _ in range(self._iterations):
            # Get a random nest
            i = random.randint(0, self._population_size - 1)
            cuckoo_nest = self._nests[i]
            
            # Generate new solution via Levy flight
            new_position = self._get_cuckoo(cuckoo_nest)
            
            if self.is_point_valid(new_position):
                new_fitness = self.objective_function(new_position)
                
                # Random nest to compare with
                j = random.randint(0, self._population_size - 1)
                
                # Replace if better
                if ((self._to_max and new_fitness > self._nests[j].fitness) or
                    (not self._to_max and new_fitness < self._nests[j].fitness)):
                    self._nests[j] = SolutionBasic(new_position, new_fitness)
            
            # Abandon worst nests and generate new ones
            self._abandon_nests()
            
            # Update best solution
            current_best = self.find_best_solution(self._nests)
            if ((self._to_max and current_best.fitness > self._best_solution.fitness) or
                (not self._to_max and current_best.fitness < self._best_solution.fitness)):
                self._best_solution = current_best
        
        return self._best_solution.point, self._best_solution.fitness
