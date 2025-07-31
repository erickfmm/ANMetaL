import random
import numpy as np
from ..IMetaheuristic import IMetaheuristic
from ..ISolution import SolutionBasic

class BlackHole(IMetaheuristic):
    def __init__(self, min_value, max_value, ndims, to_max=True, objective_function=None, repair_function=None, preprocess_function=None, population_size=30):
        super().__init__(to_max)
        self._min = min_value
        self._max = max_value
        self._ndims = ndims
        self._objective_function = objective_function
        self._repair_function = repair_function if repair_function is not None else lambda p: p
        self._preprocess_function = preprocess_function if preprocess_function is not None else lambda p: p
        self._population_size = population_size
        self._black_hole = None
        self._stars = []
        
    def initialize_population(self):
        """Initialize the population of stars randomly"""
        self._stars = []
        for _ in range(self._population_size):
            point = [random.uniform(self._min, self._max) for _ in range(self._ndims)]
            if self._repair_function:
                point = self._repair_function(point)
            if self._preprocess_function:
                point = self._preprocess_function(point)
            
            fitness = self._objective_function(point)
            if fitness is not False and fitness is not None:
                self._stars.append(SolutionBasic(point, fitness))
        
        # Find the initial black hole (best solution)
        if self._stars:
            self._black_hole = self.find_best_solution(self._stars)
        
    def run_yielded(self, iterations: int = 100, population: int = 30, seed: int = None, verbose: bool = False):
        """Execute the Black Hole algorithm with yielding"""
        self._iterations = iterations
        self._population = population
        self._seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.initialize_population()
        
        if not self._stars or not self._black_hole:
            return
            
        iteration = 1
        best_solution_historical = self._black_hole
        best_fitness_historical = best_solution_historical.fitness
        best_point_historical = np.copy(best_solution_historical.point)
        
        # yield initial state
        points = [e.point for e in self._stars]
        fts = [e.fitness for e in self._stars]
        bin_point = self._preprocess_function(best_point_historical)
        yield iteration, best_fitness_historical, bin_point, points, fts
            
        for it in range(iterations):
            if verbose:
                print("it: ", it+1, " fitness mejor: ", best_fitness_historical)
                
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
                    new_point = [random.uniform(self._min, self._max) for _ in range(self._ndims)]
                    if self._repair_function:
                        new_point = self._repair_function(new_point)
                    if self._preprocess_function:
                        new_point = self._preprocess_function(new_point)
                        
                    fitness = self._objective_function(new_point)
                    if fitness is not False and fitness is not None:
                        new_stars.append(SolutionBasic(new_point, fitness))
                else:
                    # Update star position
                    if self._repair_function:
                        new_position = self._repair_function(new_position)
                    if self._preprocess_function:
                        new_position = self._preprocess_function(new_position)
                        
                    fitness = self._objective_function(new_position)
                    if fitness is not False and fitness is not None:
                        star.move_to(new_position, fitness)
                        new_stars.append(star)
                    else:
                        new_stars.append(star)
            
            self._stars = new_stars
            
            # Update black hole if better solution found
            if self._stars:
                current_best = self.find_best_solution(self._stars)
                if current_best and self._black_hole:
                    if ((self._to_max and current_best.fitness > self._black_hole.fitness) or
                        (not self._to_max and current_best.fitness < self._black_hole.fitness)):
                        self._black_hole = current_best
                        
            iteration += 1
            
            # Update best solution
            if ((self._to_max and self._black_hole.fitness > best_fitness_historical) or
                (not self._to_max and self._black_hole.fitness < best_fitness_historical)):
                best_fitness_historical = self._black_hole.fitness
                best_point_historical = np.copy(self._black_hole.point)
                
            # yield current state
            points = [e.point for e in self._stars]
            fts = [e.fitness for e in self._stars]
            bin_point = self._preprocess_function(best_point_historical)
            yield iteration, best_fitness_historical, bin_point, points, fts
    
    def run(self, iterations: int = 100, population: int = 30, seed: int = None, verbose: bool = False):
        """Execute the Black Hole algorithm"""
        for _, best_fitness_historical, bin_point, _, _ in self.run_yielded(
            iterations, population, seed, verbose):
            continue
        return best_fitness_historical, bin_point
    
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
        return [max(self._min, min(self._max, pos)) for pos in new_position]
    
    def _is_within_event_horizon(self, star_position, event_horizon):
        """Check if a star is within the event horizon of the black hole"""
        distance = np.sqrt(sum((s - b) ** 2 for s, b in zip(star_position, self._black_hole.point)))
        return distance < event_horizon
