from abc import ABC, ABCMeta, abstractmethod


class IMetaheuristic(ABC):
    __metaclass__ = ABCMeta

    def __init__(self, to_max):
        self._iterations = None
        self._population = None
        self._to_max = to_max
        self.objective_function = None
    
    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def initialize_population(self):
        raise NotImplementedError
    
    def is_point_valid(self, point):
        fitness = self.objective_function(point)
        if not fitness:
            return False
        else:
            return True

    def find_best_point(self, points: list) -> list: #all points are valid
        best = None
        best_fitness = None
        for point in points:
            fitness = self.objective_function(point)
            if best is None:
                best = point
            if best_fitness is None:
                best_fitness = fitness
            if self._to_max and fitness > best_fitness:
                best = point
                best_fitness = fitness
            if not self._to_max and fitness < best_fitness:
                best = point
                best_fitness = fitness
        return best
    
    def find_best_solution(self, solutions):
        best = None
        best_fitness = None
        for sol in solutions:
            if best is None:
                best = sol
            if best_fitness is None:
                best_fitness = sol.fitness
            if self._to_max and sol.fitness > best_fitness:
                best = sol
                best_fitness = sol.fitness
            if not self._to_max and sol.fitness < best_fitness:
                best = sol
                best_fitness = sol.fitness
        return best
    
    @staticmethod
    def cut_mod_point(point, min_x, max_x):
        newpoint = []
        ran = abs(max_x-min_x)
        for x in point:
            if x < min_x:
                dist = abs(x - min_x)
                newnum = max_x-(dist%ran)
                newpoint.append(newnum)
            elif x > max_x:
                dist = abs(max_x - x)
                newnum = min_x+(dist%ran)
                newpoint.append(newnum)
            else:
                newpoint.append(x)
        return newpoint
