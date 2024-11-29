#This file test the genetic algorithm or other metaheuristics that use categorical variables, on knapsack and maybe other categorical problems

from anmetal.population.Genetic.GeneticMH_Categorical import GeneticMH_Categorical

from anmetal.population.Genetic.GeneticMH_Categorical_WithLeap import GeneticMH_Categorical_WithLeap

from anmetal.problems.nphard_categorical.knapsack import Knapsack_Categorical

#import numpy as np
from numpy.random import RandomState
from typing import List, Callable, Tuple

print("hello it is categorical mh test")
#general parameters
knapsack_capacity:float = 50.8
total_posible_elements:int = 50

knpsk = Knapsack_Categorical(knapsack_capacity, total_posible_elements, seed=0, max_value=5, max_cost=6)

print("Genetic")
print("#"*40)
print("#"*40)
mh = GeneticMH_Categorical(categorics=knpsk.get_possible_categories(), ndims=knpsk.total_posible_elements, to_max=True, objective_function=knpsk.objective_function, repair_function=knpsk.repair_function)
fit, pt = mh.run(verbose=True)
print("fitness:")
print(fit)
print("point:")
print(pt)
print("movements:")
print(mh.movements)

knpsk.get_values_in(pt, True)

print("#"*40)
print("#"*40)
print("Genetic with leap")
mh = GeneticMH_Categorical_WithLeap(categorics=knpsk.get_possible_categories(), ndims=knpsk.total_posible_elements, to_max=True, objective_function=knpsk.objective_function, repair_function=knpsk.repair_function)
fit, pt = mh.run(verbose=True)
print("fitness:")
print(fit)
print("point:")
print(pt)
print("movements:")
print(mh.movements)

knpsk.get_values_in(pt, True)