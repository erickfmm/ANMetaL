import anmetal.utils.points_utils as utils
from anmetal.optimizer.population.IMetaheuristic import IMetaheuristic
from anmetal.optimizer.population.ISolution import SolutionWithId
import numpy as np
from numpy.random import RandomState
from typing import List, Callable

class GreedyMH_Real_WithLeap(IMetaheuristic):
    def __init__(self, min_value: float, max_value: float, ndims: int, to_max: bool,
     objective_function: Callable[[List[float]], float],
      repair_function: Callable[[List[float]], List[float]],
      preprocess_function: Callable[[List[float]], List[float]] = None):
        self._group = []
        self._random_generator : RandomState = None
        self._min = min_value
        self._max = max_value
        self._ndims = ndims
        self._to_max = to_max

        self.objective_function = objective_function
        self.preprocess_function = \
         preprocess_function if preprocess_function is not None else lambda p: p
        self.repair_function = \
         repair_function if repair_function is not None else lambda p: p


    
    def run_yielded(self, iterations: int = 100, population: int =30, seed: int = None, verbose:bool=False,
     stagnation_variation: float =0.2, its_stagnation: int =5, leap_percentage: float =0.5):
        self._iterations = iterations
        self._population = population
        self._seed = seed
        self._random_generator = RandomState(seed)

        self.initialize_population(population)
        iteration = 1
        tau = 1
        best_solution_historical = self.find_best_solution(self._group)
        best_fitness_historical = best_solution_historical.fitness
        best_point_historical = np.copy(best_solution_historical.point)
        fitness_anterior_estancado = best_fitness_historical
        #yield
        points = [e.point for e in self._group]
        fts = [e.fitness for e in self._group]
        bin_point = self.preprocess_function(best_point_historical)
        yield iteration, best_fitness_historical, bin_point, points, fts
        while iteration <= iterations:
            if verbose:
                print("it: ", iteration, " fitness mejor: ", best_fitness_historical)
            for individual in self._group:
                result_point, fitness = self.Move(individual)
                individual.move_to(result_point, fitness)# self.objective_function(self.preprocess_function(result_point)))
                # print("fitness del fish: ",fish.fish_id," es: ",fish.fitness)
            if its_stagnation is not None and iteration == tau * its_stagnation:
                fitness_mejor_actual = self.find_best_solution(self._group).fitness
                variation = np.abs(fitness_anterior_estancado - fitness_mejor_actual) / fitness_anterior_estancado
                # print("variación: ", variation)
                if variation < stagnation_variation:
                    self.Move_random(leap_percentage)
                fitness_anterior_estancado = fitness_mejor_actual
                tau += 1
            iteration += 1
            # print("seteando historicoooooooooooooooooooooooo")
            best_solution_it = self.find_best_solution(self._group)
            best_fitness_it = best_solution_it.fitness
            best_point_it = np.copy(best_solution_it.point)
            if self._to_max and best_fitness_it > best_fitness_historical:
                best_fitness_historical = best_fitness_it
                best_point_historical = best_point_it
            if not self._to_max and best_fitness_it < best_fitness_historical:
                best_fitness_historical = best_fitness_it
                best_point_historical = best_point_it
            #yield
            points = [e.point for e in self._group]
            fts = [e.fitness for e in self._group]
            bin_point = self.preprocess_function(best_point_historical)
            yield iteration, best_fitness_historical, bin_point, points, fts
        bin_point = self.preprocess_function(best_point_historical)
        # print("suma de 1s: ", np.sum(bin_point))
        #return best_fitness_historical, bin_point
        #yield
        points = [e.point for e in self._group]
        fts = [e.fitness for e in self._group]
        yield iteration, best_fitness_historical, bin_point, points, fts

    def run(self, iterations: int = 100, population: int =30, seed: int = None, verbose:bool=False,
     stagnation_variation: float =0.2, its_stagnation: int =5, leap_percentage: float =0.5):
        for _, best_fitness_historical, bin_point, _, _ in self.run_yielded(
            iterations, population, seed, verbose, stagnation_variation,
            its_stagnation, leap_percentage):
            continue
        return best_fitness_historical, bin_point
    
    def initialize_population(self, population: int):
        for indivIndex in range(0, population):
            point, fitness = self.generate_random_point()
            individual = SolutionWithId(indivIndex, point, fitness)
            self._group.append(individual)
    

    def Move(self, individual): #get closer
        origin_point = np.copy(individual.point)
        dest_point = self.find_best_solution(self._group)
        dest_point = dest_point.point
        for idim in range(len(origin_point)):
            origin_point[idim] += self._random_generator.uniform() * (origin_point[idim] - dest_point[idim])
        return self.repair_or_not(origin_point)
    
    def Move_random(self, leap_percentage): #escape
        num_to_leap = int(self._population * leap_percentage)
        index_to_leap = [i for i in range(self._population)]
        self._random_generator.shuffle(index_to_leap)
        index_to_leap = index_to_leap[0:num_to_leap]
        for sol_index in index_to_leap:
            gen_point, gen_fitness = self.generate_random_point()
            self._group[sol_index].move_to(gen_point, gen_fitness)




    def generate_random_point(self):
        cartesian_point = self._random_generator.uniform(self._min,
                                                        self._max,
                                                        self._ndims)
        return self.repair_or_not(cartesian_point)
    
    def repair_or_not(self, cartesian_point: List[float]):
        cartesian_point = IMetaheuristic.cut_mod_point(cartesian_point, self._min, self._max)
        fitness = self.objective_function(self.preprocess_function(cartesian_point))
        if not fitness:
            new_point = self.repair_function(cartesian_point)
            fitness = self.objective_function(self.preprocess_function(new_point))
            return new_point, fitness
        else:
            return cartesian_point, fitness