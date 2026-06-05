import random
import numpy as np
from ..IMetaheuristic import IMetaheuristic
from ..ISolution import SolutionBasic
from typing import List, Callable

class ArtificialBeeColony(IMetaheuristic):
    def __init__(self, min_value: float, max_value: float, ndims: int, to_max: bool,
     objective_function: Callable[[List[float]], float],
      repair_function: Callable[[List[float]], List[float]],
      preprocess_function: Callable[[List[float]], List[float]] = None):
        """
        Initialize Artificial Bee Colony Algorithm
        
        Args:
            to_max: Whether to maximize (True) or minimize (False)
            colony_size: Size of the colony (number of employed bees = number of onlooker bees)
            limit: Maximum number of trials before abandoning a food source
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

        self._food_sources = []  # food sources / solutions
        self._trials = []  # trial counter for each food source
        self._best_solution = None
    
    def initialize_population(self, population_size):
        """Initialize food sources and their trial counters"""
        self._food_sources = []
        self._trials = []
        
        for _ in range(population_size):
            point = [random.uniform(self._min, self._max) for _ in range(self._ndims)]
            point = self.cut_mod_point(point, self._min, self._max)
            # Use repair function if point is invalid
            repaired_point = self.repair_function(point)
            fitness = self.objective_function(self.preprocess_function(repaired_point))
            if fitness is not False and fitness is not None:
                self._food_sources.append(SolutionBasic(repaired_point, fitness))
                self._trials.append(0)
        
        # Ensure we have at least one valid solution
        while len(self._food_sources) < 1:
            point = [random.uniform(self._min, self._max) for _ in range(self._ndims)]
            point = self.cut_mod_point(point, self._min, self._max)
            repaired_point = self.repair_function(point)
            fitness = self.objective_function(self.preprocess_function(repaired_point))
            if fitness is not False and fitness is not None:
                self._food_sources.append(SolutionBasic(repaired_point, fitness))
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
        
        for i in range(self._ndims):
            # Generate new position component
            new_pos = (current.point[i] + 
                      phi * (current.point[i] - partner.point[i]))
            new_position.append(new_pos)
        
        return self.cut_mod_point(new_position, self._min, self._max)
    
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
            new_position = self.cut_mod_point(new_position, self._min, self._max)
            repaired_position = self.repair_function(new_position)
            new_fitness = self.objective_function(self.preprocess_function(repaired_position))
            
            # Replace if better and fitness is valid
            if (new_fitness is not False and new_fitness is not None and
                ((self._to_max and new_fitness > self._food_sources[i].fitness) or
                (not self._to_max and new_fitness < self._food_sources[i].fitness))):
                self._food_sources[i] = SolutionBasic(repaired_position, new_fitness)
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
            new_position = self.cut_mod_point(new_position, self._min, self._max)
            repaired_position = self.repair_function(new_position)
            new_fitness = self.objective_function(self.preprocess_function(repaired_position))
            
            # Replace if better and fitness is valid
            if (new_fitness is not False and new_fitness is not None and
                ((self._to_max and new_fitness > self._food_sources[selected_idx].fitness) or
                (not self._to_max and new_fitness < self._food_sources[selected_idx].fitness))):
                self._food_sources[selected_idx] = SolutionBasic(repaired_position, new_fitness)
                self._trials[selected_idx] = 0
            else:
                self._trials[selected_idx] += 1
    
    def _scout_bee_phase(self):
        """Scout bee phase"""
        for i in range(self._colony_size):
            # If trials limit exceeded, replace with new random solution
            if self._trials[i] >= self._limit:
                point = [random.uniform(self._min, self._max) for _ in range(self._ndims)]
                point = self.cut_mod_point(point, self._min, self._max)
                repaired_point = self.repair_function(point)
                fitness = self.objective_function(self.preprocess_function(repaired_point))
                if fitness is not False and fitness is not None:
                    self._food_sources[i] = SolutionBasic(repaired_point, fitness)
                    self._trials[i] = 0
    
    def run_yielded(self, iterations: int = 100, population: int = 30, limit: int = 20, seed: int = None, verbose: bool = False):
        """Execute the Artificial Bee Colony algorithm with yielding"""
        self._iterations = iterations
        self._population = population
        self._seed = seed
        self._colony_size = population // 2  # number of employed bees = number of food sources
        self._limit = limit  # limit of trials for abandonment
        
        random.seed(seed)
        np.random.seed(seed)

        self.initialize_population(self._colony_size)
        
        iteration = 1
        best_solution_historical = self.find_best_solution(self._food_sources)
        best_fitness_historical = best_solution_historical.fitness
        best_point_historical = np.copy(best_solution_historical.point)
        
        # yield initial state
        points = [e.point for e in self._food_sources]
        fts = [e.fitness for e in self._food_sources]
        bin_point = self.preprocess_function(best_point_historical)
        yield iteration, best_fitness_historical, bin_point, points, fts
        
        while iteration <= iterations:
            if verbose:
                print("it: ", iteration, " fitness mejor: ", best_fitness_historical)
                
            # Employed bee phase
            self._employed_bee_phase()
            
            # Onlooker bee phase
            self._onlooker_bee_phase()
            
            # Scout bee phase
            self._scout_bee_phase()
            
            iteration += 1
            
            # Update best solution
            current_best = self.find_best_solution(self._food_sources)
            if ((self._to_max and current_best.fitness > best_fitness_historical) or
                (not self._to_max and current_best.fitness < best_fitness_historical)):
                best_fitness_historical = current_best.fitness
                best_point_historical = np.copy(current_best.point)
                
            # yield current state
            points = [e.point for e in self._food_sources]
            fts = [e.fitness for e in self._food_sources]
            bin_point = self.preprocess_function(best_point_historical)
            yield iteration, best_fitness_historical, bin_point, points, fts
        
        # Final yield
        bin_point = self.preprocess_function(best_point_historical)
        points = [e.point for e in self._food_sources]
        fts = [e.fitness for e in self._food_sources]
        yield iteration, best_fitness_historical, bin_point, points, fts

    def run(self, iterations: int = 100, population: int = 30, limit: int = 20, seed: int = None, verbose: bool = False):
        """Execute the Artificial Bee Colony algorithm"""
        for _, best_fitness_historical, bin_point, _, _ in self.run_yielded(
            iterations, population, limit, seed, verbose):
            continue
        return best_fitness_historical, bin_point
