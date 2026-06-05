import anmetal.utils.points_utils as utils
from anmetal.optimizer.population.IMetaheuristic import IMetaheuristic
from anmetal.optimizer.population.ISolution import SolutionWithId
import numpy as np
from numpy.random import RandomState
from typing import List, Callable

class AFSAMH_Real(IMetaheuristic):
    def __init__(self, min_value: float, max_value: float, ndims: int, to_max: bool,
     objective_function: Callable[[List[float]], float],
      repair_function: Callable[[List[float]], List[float]],
      preprocess_function: Callable[[List[float]], List[float]] = None):
        self._swarm = []
        self._random_generator : RandomState = None
        
        self._min = min_value
        self._max = max_value
        self._ndims = ndims
        self._to_max = to_max
        self.veces_movimiento = dict(move=0, prey=0, follow=0, swarm=0, leap=0)

        self.objective_function = objective_function
        self.preprocess_function = \
         preprocess_function if preprocess_function is not None else lambda p: p
        self.repair_function = \
         repair_function if repair_function is not None else lambda p: p


    
    def run_yielded(self, iterations: int = 100, population: int =30, verbose:bool=False,
     stagnation_variation: float =0.2, its_stagnation: int =5, leap_percentage: float =0.5, \
      velocity_percentage: float =0.3, n_points_to_choose: int =1, crowded_percentage: float =0.9,\
       visual_distance_percentage: float =0.1, seed: int = None):
        self._iterations = iterations
        self._population = population
        self._seed = seed
        self._visual_distance = np.sqrt(pow(self._max, 2) * self._ndims) * visual_distance_percentage
        self.n_points_to_choose = n_points_to_choose
        self._velocity_percentage = velocity_percentage
        self._random_generator = RandomState(seed)

        self.initialize_population(population)
        iteration = 1
        tau = 1
        best_solution_historical = self.find_best_solution(self._swarm)
        best_fitness_historical = best_solution_historical.fitness
        best_point_historical = np.copy(best_solution_historical.point)
        fitness_anterior_estancado = best_fitness_historical
        #yield
        points = [e.point for e in self._swarm]
        fts = [e.fitness for e in self._swarm]
        bin_point = self.preprocess_function(best_point_historical)
        yield iteration, best_fitness_historical, bin_point, points, fts
        while iteration <= iterations:
            if verbose:
                print("it: ", iteration, " fitness mejor: ", best_fitness_historical)
            for fish in self._swarm:
                neighborhood, neigh_points = self.get_neighborhood(fish)
                if len(neigh_points) == 0:
                    result_point = self.AF_Move(fish)
                else:
                    if len(neigh_points) / len(self._swarm) >= crowded_percentage:
                        result_point = self.AF_Follow(fish, neighborhood)
                    else:
                        result_point1 = self.AF_Swarm(fish, neighborhood)
                        result_point2 = self.AF_Prey(fish, neighborhood)
                        result_point = self.find_best_point([result_point1, result_point2])
                #result_point = self.borrar_1s_extra(result_point)
                result_point, fitness = self.repair_or_not(result_point)
                fish.move_to(result_point, fitness)# self.objective_function(self.preprocess_function(result_point)))
                # print("fitness del fish: ",fish.fish_id," es: ",fish.fitness)
            if iteration == tau * its_stagnation:
                fitness_mejor_actual = self.find_best_solution(self._swarm).fitness
                variation = 0
                if fitness_anterior_estancado != 0: #to avoid dividing by zero
                    variation = np.abs(fitness_anterior_estancado - fitness_mejor_actual) / fitness_anterior_estancado
                # print("variación: ", variation)
                if variation < stagnation_variation:
                    self.AF_Leap(leap_percentage)
                fitness_anterior_estancado = fitness_mejor_actual
                tau += 1
            iteration += 1
            # print("seteando historicoooooooooooooooooooooooo")
            best_solution_it = self.find_best_solution(self._swarm)
            best_fitness_it = best_solution_it.fitness
            best_point_it = np.copy(best_solution_it.point)
            if self._to_max and best_fitness_it > best_fitness_historical:
                best_fitness_historical = best_fitness_it
                best_point_historical = best_point_it
            if not self._to_max and best_fitness_it < best_fitness_historical:
                best_fitness_historical = best_fitness_it
                best_point_historical = best_point_it
            #yield
            points = [e.point for e in self._swarm]
            fts = [e.fitness for e in self._swarm]
            bin_point = self.preprocess_function(best_point_historical)
            yield iteration, best_fitness_historical, bin_point, points, fts
        # print("veces por movimiento: ", self.veces_movimiento)
        # print("terminó: ", mejor_fitness_historico)
        #yield
        points = [e.point for e in self._swarm]
        fts = [e.fitness for e in self._swarm]
        bin_point = self.preprocess_function(best_point_historical)
        yield iteration, best_fitness_historical, bin_point, points, fts

    def run(self, iterations: int = 100, population: int =30, verbose:bool=False,
     stagnation_variation: float =0.2, its_stagnation: int =5, leap_percentage: float =0.5, \
      velocity_percentage: float =0.3, n_points_to_choose: int =1, crowded_percentage: float =0.9,\
       visual_distance_percentage: float =0.1, seed: int = None):
        for _, best_fitness_historical, bin_point, _, _ in self.run_yielded(
            iterations, population, verbose, stagnation_variation, its_stagnation,
            leap_percentage, velocity_percentage, n_points_to_choose, crowded_percentage,
            visual_distance_percentage, seed):
            continue
        return best_fitness_historical, bin_point
    
    def initialize_population(self, population: int):
        for fishIndex in range(0, population):
            point, fitness = self.generate_random_point()
            fish = SolutionWithId(fishIndex, point, fitness)
            self._swarm.append(fish)



    def AF_Move(self, fish: SolutionWithId): #move randomly in visual distance
        self.veces_movimiento["move"] += 1
        points = []
        for _ in range(self.n_points_to_choose):
            point = self.generate_in_visual(fish.point)
            points.append(point)
        return self.find_best_point(points)

    def AF_Swarm(self, fish: SolutionWithId, neighborhood: List[SolutionWithId]): #move to center of neighborhood
        self.veces_movimiento["swarm"] += 1
        central_point = self.get_central_point(neighborhood)
        if self.is_point_valid(central_point) \
         and np.array_equal(self.find_best_point([central_point, fish.point]), central_point):
            return central_point
        else:
            return self.AF_Follow(fish, neighborhood)

    def AF_Prey(self, fish: SolutionWithId, neighborhood: List[SolutionWithId]): #move to best neighbour
        self.veces_movimiento["prey"] +=1
        # neigh_fishes, neigh_points = self.get_neighborhood(fish)
        best_neighbour = self.find_best_solution(neighborhood)
        if self.find_best_solution([best_neighbour, fish]) == fish: # == best_neighbour: #the original is to best_neigh
            return self.get_closer_to(fish.point, best_neighbour.point)
        else:
            return self.AF_Follow(fish, neighborhood)

    def AF_Follow(self, fish: SolutionWithId, neighborhood: List[SolutionWithId]): #move to any neighbour
        self.veces_movimiento["follow"] += 1
        # neighFishes, neighPoints = self.get_neighborhood(fish)
        points = []
        for _ in range(self.n_points_to_choose):
            neighbor_to = int(self._random_generator.uniform(0, len(neighborhood) - 1))
            points.append(neighborhood[neighbor_to].point)
        return self.get_closer_to(fish.point, self.find_best_point(points))

    def AF_Leap(self, leap_percentage: float): #escape
        self.veces_movimiento["leap"] += 1
        num_fishes_to_leap = int(self._population * leap_percentage)
        index_fishes_to_leap = [i for i in range(self._population)]
        self._random_generator.shuffle(index_fishes_to_leap)
        index_fishes_to_leap = index_fishes_to_leap[0:num_fishes_to_leap]
        for fish_index in index_fishes_to_leap:
            # fish_index = int(np.floor(config.random_generator.uniform(0, self.config["fishNumber"])))
            gen_point, gen_fitness = self.generate_random_point()
            self._swarm[fish_index].move_to(gen_point, gen_fitness)


    def get_neighborhood(self, fish: SolutionWithId):
        neighborhood = []
        points_neighborhood = []
        for f in self._swarm:
            if f._id != fish._id:
                distance = utils.distance(fish.point, f.point)
                if distance <= self._visual_distance:
                    neighborhood.append(f)
                    points_neighborhood.append(f.point)
        return neighborhood, points_neighborhood

    def get_central_point(self, neighborhood: List[SolutionWithId]):
        media_dim = []
        for i in range(0, self._ndims):
            media_dim.append(0)
        # neigh_fishes, neigh_points = self.get_neighborhood(fish)
        for neighbor in neighborhood:
            p = neighbor.point
            for i in range(0, len(media_dim)):
                media_dim[i] += p[i]
        for i in range(0, len(media_dim)):
            media_dim[i] /= len(neighborhood)
        return self.repair_or_not(media_dim)[0]

    def is_in_visual(self, fish: SolutionWithId, point: List[float]):
        distance = utils.distance(fish.point, point)
        if distance <= self._visual_distance:
            return True
        else:
            return False

    def generate_in_visual(self, origin_point: List[float]):
        distance = self._random_generator.uniform(0 , self._visual_distance, 1)
        angles = self._random_generator.uniform(0 , 2*np.pi, self._ndims -1)
        cartesian_point = utils.nsphere_to_cartesian(distance, angles)
        for i in range(0, len(cartesian_point)):
            cartesian_point[i] += origin_point[i] #scale
        return self.repair_or_not(cartesian_point)[0]

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
    
    def get_closer_to(self, origin_point: List[float], dest_point: List[float]):
        for idim in range(len(origin_point)):
            origin_point[idim] += self._velocity_percentage * (origin_point[idim] - dest_point[idim])
        return self.repair_or_not(origin_point)[0]