import random
import numpy as np
from ..IMetaheuristic import IMetaheuristic
from ..ISolution import SolutionBasic

class HarmonySearch(IMetaheuristic):
    def __init__(self, to_max=True, hms=30, hmcr=0.9, par=0.3, bw=0.01):
        """
        Initialize Harmony Search Algorithm
        
        Args:
            to_max: Whether to maximize (True) or minimize (False)
            hms: Harmony Memory Size
            hmcr: Harmony Memory Considering Rate
            par: Pitch Adjustment Rate
            bw: Bandwidth
        """
        super().__init__(to_max)
        self._hms = hms  # harmony memory size
        self._hmcr = hmcr  # harmony memory considering rate
        self._par = par  # pitch adjustment rate
        self._bw = bw  # bandwidth
        self._harmony_memory = []
        self._best_solution = None
    
    def initialize_population(self):
        """Initialize the harmony memory"""
        self._harmony_memory = []
        for _ in range(self._hms):
            point = [random.uniform(self.min_x, self.max_x) for _ in range(self.dimension)]
            if self.is_point_valid(point):
                fitness = self.objective_function(point)
                self._harmony_memory.append(SolutionBasic(point, fitness))
        
        self._best_solution = self.find_best_solution(self._harmony_memory)
    
    def _memory_consideration(self, dimension):
        """Select a value from harmony memory"""
        random_harmony = random.choice(self._harmony_memory)
        return random_harmony.point[dimension]
    
    def _pitch_adjustment(self, value):
        """Adjust the pitch of a note"""
        if random.random() < self._par:
            return value + self._bw * random.uniform(-1, 1)
        return value
    
    def _create_new_harmony(self):
        """Generate a new harmony"""
        new_harmony = []
        for i in range(self.dimension):
            if random.random() < self._hmcr:
                # Memory consideration
                value = self._memory_consideration(i)
                # Pitch adjustment
                value = self._pitch_adjustment(value)
            else:
                # Random selection
                value = random.uniform(self.min_x, self.max_x)
            new_harmony.append(value)
        
        return self.cut_mod_point(new_harmony, self.min_x, self.max_x)
    
    def _update_harmony_memory(self, new_harmony, new_fitness):
        """Update harmony memory if new harmony is better than worst harmony"""
        worst_index = 0
        worst_fitness = self._harmony_memory[0].fitness
        
        for i in range(1, self._hms):
            if ((self._to_max and self._harmony_memory[i].fitness < worst_fitness) or
                (not self._to_max and self._harmony_memory[i].fitness > worst_fitness)):
                worst_index = i
                worst_fitness = self._harmony_memory[i].fitness
        
        if ((self._to_max and new_fitness > worst_fitness) or
            (not self._to_max and new_fitness < worst_fitness)):
            self._harmony_memory[worst_index] = SolutionBasic(new_harmony, new_fitness)
    
    def run(self):
        """Execute the Harmony Search algorithm"""
        self.initialize_population()
        
        for _ in range(self._iterations):
            # Create new harmony
            new_harmony = self._create_new_harmony()
            
            # Evaluate new harmony
            if self.is_point_valid(new_harmony):
                new_fitness = self.objective_function(new_harmony)
                
                # Update harmony memory
                self._update_harmony_memory(new_harmony, new_fitness)
                
                # Update best solution
                current_best = self.find_best_solution(self._harmony_memory)
                if ((self._to_max and current_best.fitness > self._best_solution.fitness) or
                    (not self._to_max and current_best.fitness < self._best_solution.fitness)):
                    self._best_solution = current_best
        
        return self._best_solution.point, self._best_solution.fitness
