#This file test the Real version of the Metaheuristics (Numerical) with the problems in anmetal.problems.nphard_real package that works on 2D vector points, creaing a figure in each iteration with the position of each point and the position of the just past iteration with a line between them for each point. to view the evolution and movement of the solution candidates (points) in the search space

from anmetal.population.AFSA.AFSAMH_Real import AFSAMH_Real
from anmetal.population.SillyRandom.GreedyMH_Real import GreedyMH_Real
from anmetal.population.SillyRandom.GreedyMH_Real_WithLeap import GreedyMH_Real_WithLeap
from anmetal.population.PSO.PSOMH_Real import PSOMH_Real
from anmetal.population.PSO.PSOMH_Real_WithLeap import PSOMH_Real_WithLeap


#from problems.nphard_real.partition__and_subset_sum import Partition_Real, Subset_Real
import anmetal.problems.nonlinear_functions.two_inputs as problems_2

import matplotlib.pyplot as plt
from numpy.random import RandomState
import numpy as np
import os
from os.path import exists, join

import argparse
import shutil

parser = argparse.ArgumentParser(description='Metaheuristic image graph maker')
#parser.add_argument("--a", default=1, type=int, help="This is the 'a' variable")

parser.add_argument("--seed", default=0, type=int, help="The integer to be seed of random number generator")
parser.add_argument("--mh", default="", type=str, help="\
    Name of Metaheuristic, can be of of:\n\
    AFSA\n\
    PSO\n\
    PSOWL\n\
    Greed\n\
    GreedWL")
parser.add_argument("--problem", default="Goldsteinprice", type=str, help="\
    Name of the problem, can be one of:\
    Camelback\n\
    Goldsteinprice\n\
    Pshubert1\n\
    Pshubert2\n\
    Shubert\n\
    Quartic")
parser.add_argument("--verbose", default=1, type=int, help="1 if print logs, 0 if not print")
parser.add_argument("--iterations", default=100, type=int, help="Number of iterations in Metaheuristic")
parser.add_argument("--population", default=30, type=int, help="Number of solutions in Metaheuristic")

args = parser.parse_args()

print("args: ", args)

to_verbose : bool = True if args.verbose == 1 else False
seed: int = args.seed
random_generator = RandomState(seed)
iterations = args.iterations
population = args.population

probs_dict = {
    "Camelback": problems_2.Camelback,
    "Goldsteinprice": problems_2.Goldsteinprice,
    "Pshubert1": problems_2.Pshubert1,
    "Pshubert2": problems_2.Pshubert2,
    "Shubert": problems_2.Shubert,
    "Quartic": problems_2.Quartic
}

prob = probs_dict[args.problem] if args.problem in probs_dict else problems_2.Camelback
#partition_problem = Partition_Real(seed=seed, num_dims=200)

if str.lower(args.mh) == "afsa" or args.mh == "":
    mh = AFSAMH_Real(prob.get_limits()[0], prob.get_limits()[1], 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, visual_distance_percentage=0.5, velocity_percentage=0.5, n_points_to_choose=3, crowded_percentage=0.7, its_stagnation=4, leap_percentage=0.3, stagnation_variation=0.4, seed=seed)
elif str.lower(args.mh) == "pso":
    mh = PSOMH_Real(prob.get_limits()[0], prob.get_limits()[1], 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, omega=0.8, phi_g=1, phi_p=0.5)
elif str.lower(args.mh) == "psowl":
    mh = PSOMH_Real_WithLeap(prob.get_limits()[0], prob.get_limits()[1], 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, omega=0.8, phi_g=1, phi_p=0.5)
elif str.lower(args.mh) == "greed":
    mh = GreedyMH_Real(prob.get_limits()[0], prob.get_limits()[1], 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, stagnation_variation=0.4, its_stagnation=5, leap_percentage=0.8)
elif str.lower(args.mh) == "greedwl":
    mh = GreedyMH_Real(prob.get_limits()[0], prob.get_limits()[1], 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, stagnation_variation=0.4, its_stagnation=5, leap_percentage=0.8)

folderpath = join("mh_graphs", args.mh+"_"+args.problem)
if exists(folderpath):
    shutil.rmtree(folderpath, ignore_errors=True)
os.makedirs(folderpath)



print("to start iterations")
colors_to_use = []
is_first : bool = True
for iteration, best_fitness_historical, best_bin_point, points_a, fts in gen:
    print("iteration: ", iteration)
    plt.xlim(prob.get_limits()[0], prob.get_limits()[1])
    plt.ylim(prob.get_limits()[0], prob.get_limits()[1])
    ps_ = np.copy(points_a)
    if is_first: #is_first para no depender del numero
        for i in range(len(ps_)):
            color = (random_generator.uniform(0.3, 1),
                    random_generator.uniform(0.3, 1),
                    random_generator.uniform(0.3, 1))
            colors_to_use.append(color)
            plt.plot(ps_[i][0], ps_[i][1], '*', color=color)
        last_points = np.copy(ps_) #copiar array
        is_first = False
    else:
        #line between points (2,5) -> (4,8)
        for i in range(len(ps_)):
            plt.plot([last_points[i][0], ps_[i][0]], [last_points[i][1], ps_[i][1]], '-', color=colors_to_use[i])
            plt.plot(ps_[i][0], ps_[i][1], '*', color=colors_to_use[i])
        last_points = np.copy(ps_) #copiar array
    plt.savefig(join(folderpath, "mhgraph_"+str(iteration)+".png"))
    plt.clf()