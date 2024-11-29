import random
import numpy as np
from ..IMetaheuristic import IMetaheuristic
from ..ISolution import SolutionBasic

class BlackHole(IMetaheuristic):
    def __init__(self, to_max=True, population_size=30):
        super().__init__(to_max)
        self._population_size = population_size
        self._black_hole = None
        self._stars = []
        
    def initialize_population(self):
        """Initialize the population of stars randomly"""
        self._stars = []
        for _ in range(self._population_size):
            point = [random.uniform(self.min_x, self.max_x) for _ in range(self.dimension)]
            if self.is_point_valid(point):
                fitness = self.objective_function(point)
                self._stars.append(SolutionBasic(point, fitness))
        
        # Find the initial black hole (best solution)
        self._black_hole = self.find_best_solution(self._stars)
        
    def run(self):
        """Execute the Black Hole algorithm"""
        self.initialize_population()
        
        for _ in range(self._iterations):
            # Calculate the radius of event horizon
            event_horizon = self._calculate_event_horizon()
            
            # Move stars towards black hole and check for absorption
            new_stars = []
            for star in self._stars:
                if star == self._black_hole:
                    continue
                    
                # Move star towards black hole
                new_position = self._move_star_towards_black_hole(star.point)
                
                # Check if star crosses event horizon
                if self._is_within_event_horizon(new_position, event_horizon):
                    # Generate new random star
                    new_point = [random.uniform(self.min_x, self.max_x) for _ in range(self.dimension)]
                    if self.is_point_valid(new_point):
                        fitness = self.objective_function(new_point)
                        new_stars.append(SolutionBasic(new_point, fitness))
                else:
                    # Update star position if valid
                    if self.is_point_valid(new_position):
                        fitness = self.objective_function(new_position)
                        star.move_to(new_position, fitness)
                        new_stars.append(star)
                    else:
                        new_stars.append(star)
            
            self._stars = new_stars
            
            # Update black hole if better solution found
            current_best = self.find_best_solution(self._stars)
            if ((self._to_max and current_best.fitness > self._black_hole.fitness) or
                (not self._to_max and current_best.fitness < self._black_hole.fitness)):
                self._black_hole = current_best
                
        return self._black_hole.point, self._black_hole.fitness
    
    def _calculate_event_horizon(self):
        """Calculate the radius of the event horizon"""
        black_hole_fitness = abs(self._black_hole.fitness)
        total_fitness = sum(abs(star.fitness) for star in self._stars)
        return black_hole_fitness / total_fitness
    
    def _move_star_towards_black_hole(self, star_position):
        """Move a star towards the black hole"""
        new_position = []
        for i in range(len(star_position)):
            # Random number between 0 and 1
            r = random.random()
            # Move star towards black hole
            pos = star_position[i] + r * (self._black_hole.point[i] - star_position[i])
            new_position.append(pos)
        
        # Apply boundary conditions
        return self.cut_mod_point(new_position, self.min_x, self.max_x)
    
    def _is_within_event_horizon(self, star_position, event_horizon):
        """Check if a star is within the event horizon of the black hole"""
        distance = np.sqrt(sum((s - b) ** 2 for s, b in zip(star_position, self._black_hole.point)))
        return distance < event_horizon
