import random
import numpy as np
from ..IMetaheuristic import IMetaheuristic
from ..ISolution import SolutionBasic

class ArtificialBeeColony(IMetaheuristic):
    def __init__(self, to_max=True, colony_size=40, limit=20):
        """
        Initialize Artificial Bee Colony Algorithm
        
        Args:
            to_max: Whether to maximize (True) or minimize (False)
            colony_size: Size of the colony (number of employed bees = number of onlooker bees)
            limit: Maximum number of trials before abandoning a food source
        """
        super().__init__(to_max)
        self._colony_size = colony_size  # number of employed bees = number of food sources
        self._limit = limit  # limit of trials for abandonment
        self._food_sources = []  # food sources / solutions
        self._trials = []  # trial counter for each food source
        self._best_solution = None
    
    def initialize_population(self):
        """Initialize food sources and their trial counters"""
        self._food_sources = []
        self._trials = []
        
        for _ in range(self._colony_size):
            point = [random.uniform(self.min_x, self.max_x) for _ in range(self.dimension)]
            if self.is_point_valid(point):
                fitness = self.objective_function(point)
                self._food_sources.append(SolutionBasic(point, fitness))
                self._trials.append(0)
        
        self._best_solution = self.find_best_solution(self._food_sources)
    
    def _calculate_probability(self, solution):
        """Calculate probability for onlooker bee selection"""
        if self._to_max:
            return 0.9 * solution.fitness / self._best_solution.fitness + 0.1
        else:
            return 1.0 / (1.0 + abs(solution.fitness))
    
    def _generate_new_position(self, current, partner):
        """Generate new food source position"""
        new_position = []
        phi = random.uniform(-1, 1)
        
        for i in range(self.dimension):
            # Generate new position component
            new_pos = (current.point[i] + 
                      phi * (current.point[i] - partner.point[i]))
            new_position.append(new_pos)
        
        return self.cut_mod_point(new_position, self.min_x, self.max_x)
    
    def _employed_bee_phase(self):
        """Employed bee phase"""
        for i in range(self._colony_size):
            # Select random partner, different from current bee
            partner_idx = i
            while partner_idx == i:
                partner_idx = random.randint(0, self._colony_size - 1)
            
            # Generate new food source position
            new_position = self._generate_new_position(
                self._food_sources[i], 
                self._food_sources[partner_idx]
            )
            
            # Evaluate new position
            if self.is_point_valid(new_position):
                new_fitness = self.objective_function(new_position)
                
                # Replace if better
                if ((self._to_max and new_fitness > self._food_sources[i].fitness) or
                    (not self._to_max and new_fitness < self._food_sources[i].fitness)):
                    self._food_sources[i] = SolutionBasic(new_position, new_fitness)
                    self._trials[i] = 0
                else:
                    self._trials[i] += 1
    
    def _onlooker_bee_phase(self):
        """Onlooker bee phase"""
        # Calculate selection probabilities
        probabilities = [self._calculate_probability(source) for source in self._food_sources]
        prob_sum = sum(probabilities)
        probabilities = [p/prob_sum for p in probabilities]
        
        # For each onlooker bee
        for _ in range(self._colony_size):
            # Select food source based on probability
            selected_idx = np.random.choice(self._colony_size, p=probabilities)
            
            # Select random partner
            partner_idx = selected_idx
            while partner_idx == selected_idx:
                partner_idx = random.randint(0, self._colony_size - 1)
            
            # Generate new food source position
            new_position = self._generate_new_position(
                self._food_sources[selected_idx], 
                self._food_sources[partner_idx]
            )
            
            # Evaluate new position
            if self.is_point_valid(new_position):
                new_fitness = self.objective_function(new_position)
                
                # Replace if better
                if ((self._to_max and new_fitness > self._food_sources[selected_idx].fitness) or
                    (not self._to_max and new_fitness < self._food_sources[selected_idx].fitness)):
                    self._food_sources[selected_idx] = SolutionBasic(new_position, new_fitness)
                    self._trials[selected_idx] = 0
                else:
                    self._trials[selected_idx] += 1
    
    def _scout_bee_phase(self):
        """Scout bee phase"""
        for i in range(self._colony_size):
            # If trials limit exceeded, replace with new random solution
            if self._trials[i] >= self._limit:
                point = [random.uniform(self.min_x, self.max_x) for _ in range(self.dimension)]
                if self.is_point_valid(point):
                    fitness = self.objective_function(point)
                    self._food_sources[i] = SolutionBasic(point, fitness)
                    self._trials[i] = 0
    
    def run(self):
        """Execute the Artificial Bee Colony algorithm"""
        self.initialize_population()
        
        for _ in range(self._iterations):
            # Employed bee phase
            self._employed_bee_phase()
            
            # Onlooker bee phase
            self._onlooker_bee_phase()
            
            # Scout bee phase
            self._scout_bee_phase()
            
            # Update best solution
            current_best = self.find_best_solution(self._food_sources)
            if ((self._to_max and current_best.fitness > self._best_solution.fitness) or
                (not self._to_max and current_best.fitness < self._best_solution.fitness)):
                self._best_solution = current_best
        
        return self._best_solution.point, self._best_solution.fitness
