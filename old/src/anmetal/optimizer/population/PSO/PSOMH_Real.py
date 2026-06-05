import anmetal.utils.points_utils as utils
from anmetal.optimizer.population.IMetaheuristic import IMetaheuristic
from anmetal.optimizer.population.PSO.Particle import SolutionParticle
import numpy as np
from numpy.random import RandomState
from typing import List, Callable

class PSOMH_Real(IMetaheuristic):
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


    
    def run_yielded(self, iterations: int = 100, population: int =30, seed: int = None,
        omega: float=0.5, phi_p: float=1, phi_g: float=1, verbose:bool=False):
        self._iterations = iterations
        self._population = population
        self._seed = seed
        self._random_generator = RandomState(seed)

        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g

        self.initialize_population(population)
        iteration = 1
        best_solution_historical = self.find_best_solution(self._group)
        best_fitness_historical = best_solution_historical.fitness
        best_point_historical = np.copy(best_solution_historical.point)
        #yield
        points = [e.point for e in self._group]
        fts = [e.fitness for e in self._group]
        bin_point = self.preprocess_function(best_point_historical)
        yield iteration, best_fitness_historical, bin_point, points, fts
        while iteration <= iterations:
            if verbose:
                print("it: ", iteration, " fitness mejor: ", best_fitness_historical)
            for individual in self._group:
                #update the velocity
                for idim in range(len(individual.point)):
                    rp = self._random_generator.uniform()
                    rg = self._random_generator.uniform()
                    individual.velocity[idim] = omega * individual.velocity[idim] + \
                        phi_p*rp * (individual.best_point[idim] - individual.point[idim]) +\
                            phi_g*rg * (best_point_historical[idim] - individual.point[idim])
                result_point, fitness = self.Move(individual)
                individual.move_to(result_point, fitness)# self.objective_function(self.preprocess_function(result_point)))
                if self._to_max and individual.fitness > individual.best_fitness:
                    individual.set_best_point(result_point, fitness)
                if not self._to_max and individual.fitness < individual.best_fitness:
                    individual.set_best_point(result_point, fitness)
                # print("fitness del fish: ",fish.fish_id," es: ",fish.fitness)
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
    
    def run(self, iterations: int = 100, population: int =30, seed: int = None,
        omega: float=0.5, phi_p: float=1, phi_g: float=1, verbose:bool=False):
        for _, best_fitness_historical, bin_point, _, _ in self.run_yielded(
            iterations, population, seed, omega, phi_p, phi_g, verbose):
            continue
        return best_fitness_historical, bin_point

    def initialize_population(self, population: int):
        for indivIndex in range(0, population):
            point, fitness = self.generate_random_point()
            velocity = self._random_generator.uniform(-(self._max - self._min),
                                                        (self._max - self._min),
                                                        self._ndims)
            individual = SolutionParticle(indivIndex, point, fitness, velocity)
            self._group.append(individual)

    def Move(self, individual): #get closer
        origin_point = np.copy(individual.point)
        for idim in range(len(origin_point)):
            origin_point[idim] += individual.velocity[idim]
        return self.repair_or_not(origin_point)

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
    
    