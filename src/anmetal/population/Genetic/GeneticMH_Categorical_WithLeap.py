from anmetal.population.IMetaheuristic import IMetaheuristic
from anmetal.population.ISolution import SolutionWithId
import numpy as np
from numpy.random import RandomState
from typing import List, Callable, Tuple

class GeneticMH_Categorical_WithLeap(IMetaheuristic):
    def __init__(self, categorics: list, ndims: int, to_max: bool,
     objective_function: Callable[[list], float],
      repair_function: Callable[[list], list],
      preprocess_function: Callable[[list], list] = None):
        self._group: List[SolutionWithId] = []
        self._random_generator: RandomState = None
        
        self._categorics: list = categorics
        self._ndims: int = ndims
        self._to_max: bool = to_max

        self.objective_function = objective_function
        self.preprocess_function = \
         preprocess_function if preprocess_function is not None else lambda p: p
        self.repair_function = \
         repair_function if repair_function is not None else lambda p: p

    def run_yielded(self, iterations: int = 100, population: int =30,
        elitist_percentage: float = 0.3, mutability: float = 0.1, fidelity: bool = True,
        mutation_in_parents: bool = True, seed: int = None, verbose: bool = False,
        stagnation_variation: float =0.2, its_stagnation: int =5, leap_percentage: float =0.5):
        self._iterations = iterations
        self._population = population
        self._seed: int = seed

        self._mutability = mutability
        self._fidelity = fidelity
        self._mutation_in_parents = mutation_in_parents
        self.movements = {
            "random": 0,
            "init": 0,
            "select_elite": 0,
            "recombination": 0,
            "mutation": 0,
            "crossing": 0
        }
        
        self._random_generator: RandomState = RandomState(seed)

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
                print("it: ", iteration, " best historical fitness: ", best_fitness_historical)
            elite = self.select_elite(elitist_percentage)
            self.recombination(elite)
            if not self._mutation_in_parents:
                for individual in self._group:
                    individual = self.mutate_individual(individual.point)
            if its_stagnation is not None and  iteration == tau * its_stagnation:
                fitness_mejor_actual = self.find_best_solution(self._group).fitness
                variation = np.abs(fitness_anterior_estancado - fitness_mejor_actual) / fitness_anterior_estancado
                # print("variación: ", variation)
                if variation < stagnation_variation:
                    self.Move_random(leap_percentage)
                fitness_anterior_estancado = fitness_mejor_actual
                tau += 1
            iteration += 1
            # print("setting historical")
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

    def run(self, iterations: int = 100, population: int =30,
        elitist_percentage: float = 0.3, mutability: float = 0.1, fidelity: bool = True,
        mutation_in_parents: bool = True, seed: int = None, verbose: bool = False,
        stagnation_variation: float =0.2, its_stagnation: int =5, leap_percentage: float =0.5):
        for _, best_fitness_historical, bin_point, _, _ in self.run_yielded(
            iterations, population, elitist_percentage, mutability, fidelity, 
            mutation_in_parents, seed, verbose, stagnation_variation, 
            its_stagnation, leap_percentage):
            continue
        return best_fitness_historical, bin_point
    
    def initialize_population(self, population: int):
        self.movements["init"] += 1
        self._group: List[SolutionWithId] = []
        for indivIndex in range(0, population):
            point, fitness = self.generate_random_point()
            individual = SolutionWithId(indivIndex, point, fitness)
            self._group.append(individual)

    def sort_group(self):
        fitnesses: List[float] = []
        for individual in self._group:
            fitnesses.append(individual.fitness)
        newgroup:List[SolutionWithId] = []
        for sorted_index in np.argsort(fitnesses):
            newgroup.append(self._group[sorted_index])
        self._group:List[SolutionWithId] = newgroup

    def select_elite(self, elitist_percentage: float) -> List[SolutionWithId]:
        self.movements["select_elite"] +=1
        self.sort_group()
        if int(len(self._group)*elitist_percentage) <= 0:
            raise Exception("Elitist percentage is too low")
        return self._group[:int(len(self._group)*elitist_percentage)]
    
    def recombination(self, elite: List[SolutionWithId]):
        self.movements["recombination"] += 1
        """
        I have an elite, subset of universe, cross it to complete the new universe
        without fidelity:
        - I create an array of indices [0...len(elite)]
        - I shuffle it (permutation)
        - I make a for 0 to population
        - for each individual, i search a couple and i make children
        with fidelity:
        - I create an array of indices [0...len(elite)]
        - I shuffle it (permutation)
        - I split that array int 2 (male and female, just taking the half)
        - I make a couple for each of them, coupling the indices (having in account if that individual is single)
        - I make a for 0 to population and I make children, using the same order in the shuffling and repeating
        """
        newgroup:List[SolutionWithId] = []
        indexes:List[int] = [i for i in range(len(elite))]
        self._random_generator.shuffle(indexes)
        if self._fidelity:
            indexes_len:int = int(len(indexes)/2.0)
            males:List[int] = indexes[:indexes_len]
            females:List[int] = indexes[indexes_len:indexes_len+len(males)]
        i_parent:int = 0
        for i_individual in range(self._population):
            if self._fidelity:
                point, fitness = self.cross_couple(elite[males[i_parent]], elite[females[i_parent]])
                individual = SolutionWithId(i_individual, point, fitness)
                newgroup.append(individual)
                i_parent += 1
                if i_parent >= len(males):
                    i_parent = 0
            else: #not fidelity
                #search a couple for elite[i_parent]
                couple_index:int = self._random_generator.randint(0, len(indexes)-1)
                #to avoid cross with itself
                couple_index:int = couple_index if couple_index >= i_parent else couple_index+1
                point, fitness = self.cross_couple(elite[indexes[i_parent]], elite[indexes[i_parent]])
                individual = SolutionWithId(i_individual, point, fitness)
                newgroup.append(individual)
                i_parent +=1
                if i_parent >= len(indexes):
                    i_parent = 0
        self._group:List[SolutionWithId] = newgroup


    def cross_couple(self, individual1: SolutionWithId, individual2: SolutionWithId) -> Tuple[list, float]:
        self.movements["crossing"] += 1
        individual_point1 = individual1.point
        individual_point2 = individual2.point
        if len(individual_point1) != len(individual_point2):
            raise Exception("length of points are different when crossing couples")
        if self._mutation_in_parents:
            individual_point1:list = self.mutate_individual(individual1).point
            individual_point2:list = self.mutate_individual(individual2).point
        indexes = [i%2 for i in  range(len(individual_point1))]
        self._random_generator.shuffle(indexes)
        result = []
        for iIndex in range(len(indexes)):
            if indexes[iIndex] == 0:
                result.append(individual_point1[iIndex])
            else:
                result.append(individual_point2[iIndex])
        return self.repair_or_not(result)


    def mutate_individual(self, individual: SolutionWithId):
        self.movements["mutation"] += 1
        individual_point = individual.point
        indexes_to_shuffle = [i for i in  range(len(individual_point))]
        self._random_generator.shuffle(indexes_to_shuffle)
        indexes_to_shuffle = indexes_to_shuffle[:int(len(indexes_to_shuffle)*self._mutability)]
        for index_to_shuffle in indexes_to_shuffle:
            individual_point[index_to_shuffle] = \
                 self._categorics[index_to_shuffle]\
                     [self._random_generator.randint(0, len(self._categorics[index_to_shuffle]))]
        point, fitness = self.repair_or_not(individual.point)
        return SolutionWithId(individual.get_id(), point, fitness)


    def Move_random(self, leap_percentage: float):
        self.movements["random"] +=1
        indexes:List[int] = [i for i in range(len(self._group))]
        self._random_generator.shuffle(indexes)
        indexes = indexes[:int(len(indexes)*leap_percentage)]
        for i in indexes:
            del self._group[i]
            point, fitness = self.generate_random_point()
            individual = SolutionWithId(i, point, fitness)
            self._group.append(individual)


    def generate_random_point(self):
        point = [self._categorics[i][self._random_generator.randint(0, len(self._categorics[i]))]\
                 for i in range(0, self._ndims)]
        return self.repair_or_not(point)
    
    def repair_or_not(self, point: list) -> Tuple[list, float]:
        fitness = self.objective_function(self.preprocess_function(point))
        if not fitness:
            new_point = self.repair_function(point)
            fitness = self.objective_function(self.preprocess_function(new_point))
            return new_point, fitness
        else:
            return point, fitness