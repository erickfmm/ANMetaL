#This file test the Real version of the Metaheuristics (Numerical) with the problems in anmetal.problems.nphard_real package
from anmetal.population.AFSA.AFSAMH_Real import AFSAMH_Real
from anmetal.population.SillyRandom.GreedyMH_Real import GreedyMH_Real
from anmetal.population.SillyRandom.GreedyMH_Real_WithLeap import GreedyMH_Real_WithLeap
from anmetal.population.PSO.PSOMH_Real import PSOMH_Real
from anmetal.population.PSO.PSOMH_Real_WithLeap import PSOMH_Real_WithLeap
from anmetal.population.ABC.ABCMH_Real import ABCMH_Real
from anmetal.population.AntColony.ACOMH_Real import ACOMH_Real
from anmetal.population.Bat.BatMH_Real import BatMH_Real
from anmetal.population.Blackhole.BlackholeMH_Real import BlackholeMH_Real
from anmetal.population.Cuckoo.CuckooMH_Real import CuckooMH_Real
from anmetal.population.Firefly.FireflyMH_Real import FireflyMH_Real
from anmetal.population.Genetic.GeneticMH_Real import GeneticMH_Real
from anmetal.population.HarmonySearch.HSMH_Real import HSMH_Real

from anmetal.problems.nphard_real.partition__and_subset_sum import Partition_Real, Subset_Real

to_use = [
    "AFSA",
    "Greedy",
    "GreedyWL",
    "PSO",
    "PSOWL",
    "ABC",
    "ACO",
    "BAT",
    "BH",
    "CUCKOO",
    "FIREFLY",
    "GA",
    "HS"
    ]

partition_problem = Partition_Real(seed=0, num_dims=200)
subset_problem = Subset_Real(seed=0, num_dims=200)


problem_to_solve = "partition sum"
#problem_to_solve = "subset sum"

to_verbose = True

if "AFSA" in to_use:
    print("create afsa")
    if problem_to_solve == "partition sum":
        mh = AFSAMH_Real(partition_problem.min_x, partition_problem.max_x, partition_problem.ndim, False, partition_problem.objective_function, partition_problem.repair_function, partition_problem.preprocess_function)
    if problem_to_solve == "subset sum":
        mh = AFSAMH_Real(subset_problem.min_x, subset_problem.max_x, subset_problem.ndim, False, subset_problem.objective_function, subset_problem.repair_function, subset_problem.preprocess_function)
    print("to run afsa")
    fit, pt = mh.run(verbose=to_verbose, visual_distance_percentage=0.5, velocity_percentage=0.5, n_points_to_choose=3, crowded_percentage=0.7, its_stagnation=4, leap_percentage=0.3, stagnation_variation=0.4, seed=115)
    print(fit)
    print(pt)
    if problem_to_solve == "partition sum":
        partition_problem.print_difference_subsets(pt)
    if problem_to_solve == "subset sum":
        subset_problem.print_difference_subsets(pt)

if "GreedyWL" in to_use:
    print("create GreedyWL")
    if problem_to_solve == "partition sum":
        mh = GreedyMH_Real_WithLeap(partition_problem.min_x, partition_problem.max_x, partition_problem.ndim, False, partition_problem.objective_function, partition_problem.repair_function, partition_problem.preprocess_function)
    if problem_to_solve == "subset sum":
        mh = GreedyMH_Real_WithLeap(subset_problem.min_x, subset_problem.max_x, subset_problem.ndim, False, subset_problem.objective_function, subset_problem.repair_function, subset_problem.preprocess_function)
    print("to run greedy")
    fit, pt = mh.run(verbose=to_verbose, iterations=100, population=30, stagnation_variation=0.4, its_stagnation=5, leap_percentage=0.8, seed=115)
    print(fit)
    print(pt)
    if problem_to_solve == "partition sum":
        partition_problem.print_difference_subsets(pt)
    if problem_to_solve == "subset sum":
        subset_problem.print_difference_subsets(pt)

if "Greedy" in to_use:
    print("create Greedy")
    if problem_to_solve == "partition sum":
        mh = GreedyMH_Real(partition_problem.min_x, partition_problem.max_x, partition_problem.ndim, False, partition_problem.objective_function, partition_problem.repair_function, partition_problem.preprocess_function)
    if problem_to_solve == "subset sum":
        mh = GreedyMH_Real(subset_problem.min_x, subset_problem.max_x, subset_problem.ndim, False, subset_problem.objective_function, subset_problem.repair_function, subset_problem.preprocess_function)
    print("to run greedy")
    fit, pt = mh.run(verbose=to_verbose, iterations=100, population=30, seed=115)
    print(fit)
    print(pt)
    if problem_to_solve == "partition sum":
        partition_problem.print_difference_subsets(pt)
    if problem_to_solve == "subset sum":
        subset_problem.print_difference_subsets(pt)

if "PSO" in to_use:
    print("create PSO")
    if problem_to_solve == "partition sum":
        mh = PSOMH_Real(partition_problem.min_x, partition_problem.max_x, partition_problem.ndim, False, partition_problem.objective_function, partition_problem.repair_function, partition_problem.preprocess_function)
    if problem_to_solve == "subset sum":
        mh = PSOMH_Real(subset_problem.min_x, subset_problem.max_x, subset_problem.ndim, False, subset_problem.objective_function, subset_problem.repair_function, subset_problem.preprocess_function)
    print("to run PSO")
    fit, pt = mh.run(verbose=to_verbose, iterations=100, population=30, omega=0.8, phi_g=1, phi_p=0.5,seed=115)
    print(fit)
    print(pt)
    if problem_to_solve == "partition sum":
        partition_problem.print_difference_subsets(pt)
    if problem_to_solve == "subset sum":
        subset_problem.print_difference_subsets(pt)

if "PSOWL" in to_use:
    print("create PSOWL")
    if problem_to_solve == "partition sum":
        mh = PSOMH_Real_WithLeap(partition_problem.min_x, partition_problem.max_x, partition_problem.ndim, False, partition_problem.objective_function, partition_problem.repair_function, partition_problem.preprocess_function)
    if problem_to_solve == "subset sum":
        mh = PSOMH_Real_WithLeap(subset_problem.min_x, subset_problem.max_x, subset_problem.ndim, False, subset_problem.objective_function, subset_problem.repair_function, subset_problem.preprocess_function)
    print("to run PSO")
    fit, pt = mh.run(verbose=to_verbose, iterations=100, population=30, omega=0.8, phi_g=1, phi_p=0.5 ,seed=115, stagnation_variation=0.4, its_stagnation=5, leap_percentage=0.8)
    print(fit)
    print(pt)
    if problem_to_solve == "partition sum":
        partition_problem.print_difference_subsets(pt)
    if problem_to_solve == "subset sum":
        subset_problem.print_difference_subsets(pt)

if "ABC" in to_use:
    print("create ABC")
    if problem_to_solve == "partition sum":
        mh = ABCMH_Real(partition_problem.min_x, partition_problem.max_x, partition_problem.ndim, False, partition_problem.objective_function, partition_problem.repair_function, partition_problem.preprocess_function)
    if problem_to_solve == "subset sum":
        mh = ABCMH_Real(subset_problem.min_x, subset_problem.max_x, subset_problem.ndim, False, subset_problem.objective_function, subset_problem.repair_function, subset_problem.preprocess_function)
    print("to run ABC")
    fit, pt = mh.run(verbose=to_verbose, iterations=100, population=30, limit=20, seed=115)
    print(fit)
    print(pt)
    if problem_to_solve == "partition sum":
        partition_problem.print_difference_subsets(pt)
    if problem_to_solve == "subset sum":
        subset_problem.print_difference_subsets(pt)

if "ACO" in to_use:
    print("create ACO")
    if problem_to_solve == "partition sum":
        mh = ACOMH_Real(partition_problem.min_x, partition_problem.max_x, partition_problem.ndim, False, partition_problem.objective_function, partition_problem.repair_function, partition_problem.preprocess_function)
    if problem_to_solve == "subset sum":
        mh = ACOMH_Real(subset_problem.min_x, subset_problem.max_x, subset_problem.ndim, False, subset_problem.objective_function, subset_problem.repair_function, subset_problem.preprocess_function)
    print("to run ACO")
    fit, pt = mh.run(verbose=to_verbose, iterations=100, population=30, evaporation_rate=0.1, alpha=1.0, beta=2.0, seed=115)
    print(fit)
    print(pt)
    if problem_to_solve == "partition sum":
        partition_problem.print_difference_subsets(pt)
    if problem_to_solve == "subset sum":
        subset_problem.print_difference_subsets(pt)

if "BAT" in to_use:
    print("create BAT")
    if problem_to_solve == "partition sum":
        mh = BatMH_Real(partition_problem.min_x, partition_problem.max_x, partition_problem.ndim, False, partition_problem.objective_function, partition_problem.repair_function, partition_problem.preprocess_function)
    if problem_to_solve == "subset sum":
        mh = BatMH_Real(subset_problem.min_x, subset_problem.max_x, subset_problem.ndim, False, subset_problem.objective_function, subset_problem.repair_function, subset_problem.preprocess_function)
    print("to run BAT")
    fit, pt = mh.run(verbose=to_verbose, iterations=100, population=30, fmin=0, fmax=2, A=0.9, r0=0.9, seed=115)
    print(fit)
    print(pt)
    if problem_to_solve == "partition sum":
        partition_problem.print_difference_subsets(pt)
    if problem_to_solve == "subset sum":
        subset_problem.print_difference_subsets(pt)

if "BH" in to_use:
    print("create Blackhole")
    if problem_to_solve == "partition sum":
        mh = BlackholeMH_Real(partition_problem.min_x, partition_problem.max_x, partition_problem.ndim, False, partition_problem.objective_function, partition_problem.repair_function, partition_problem.preprocess_function)
    if problem_to_solve == "subset sum":
        mh = BlackholeMH_Real(subset_problem.min_x, subset_problem.max_x, subset_problem.ndim, False, subset_problem.objective_function, subset_problem.repair_function, subset_problem.preprocess_function)
    print("to run Blackhole")
    fit, pt = mh.run(verbose=to_verbose, iterations=100, population=30, seed=115)
    print(fit)
    print(pt)
    if problem_to_solve == "partition sum":
        partition_problem.print_difference_subsets(pt)
    if problem_to_solve == "subset sum":
        subset_problem.print_difference_subsets(pt)

if "CUCKOO" in to_use:
    print("create Cuckoo")
    if problem_to_solve == "partition sum":
        mh = CuckooMH_Real(partition_problem.min_x, partition_problem.max_x, partition_problem.ndim, False, partition_problem.objective_function, partition_problem.repair_function, partition_problem.preprocess_function)
    if problem_to_solve == "subset sum":
        mh = CuckooMH_Real(subset_problem.min_x, subset_problem.max_x, subset_problem.ndim, False, subset_problem.objective_function, subset_problem.repair_function, subset_problem.preprocess_function)
    print("to run Cuckoo")
    fit, pt = mh.run(verbose=to_verbose, iterations=100, population=30, pa=0.25, seed=115)
    print(fit)
    print(pt)
    if problem_to_solve == "partition sum":
        partition_problem.print_difference_subsets(pt)
    if problem_to_solve == "subset sum":
        subset_problem.print_difference_subsets(pt)

if "FIREFLY" in to_use:
    print("create Firefly")
    if problem_to_solve == "partition sum":
        mh = FireflyMH_Real(partition_problem.min_x, partition_problem.max_x, partition_problem.ndim, False, partition_problem.objective_function, partition_problem.repair_function, partition_problem.preprocess_function)
    if problem_to_solve == "subset sum":
        mh = FireflyMH_Real(subset_problem.min_x, subset_problem.max_x, subset_problem.ndim, False, subset_problem.objective_function, subset_problem.repair_function, subset_problem.preprocess_function)
    print("to run Firefly")
    fit, pt = mh.run(verbose=to_verbose, iterations=100, population=30, alpha=0.5, beta0=1.0, gamma=1.0, seed=115)
    print(fit)
    print(pt)
    if problem_to_solve == "partition sum":
        partition_problem.print_difference_subsets(pt)
    if problem_to_solve == "subset sum":
        subset_problem.print_difference_subsets(pt)

if "GA" in to_use:
    print("create Genetic Algorithm")
    if problem_to_solve == "partition sum":
        mh = GeneticMH_Real(partition_problem.min_x, partition_problem.max_x, partition_problem.ndim, False, partition_problem.objective_function, partition_problem.repair_function, partition_problem.preprocess_function)
    if problem_to_solve == "subset sum":
        mh = GeneticMH_Real(subset_problem.min_x, subset_problem.max_x, subset_problem.ndim, False, subset_problem.objective_function, subset_problem.repair_function, subset_problem.preprocess_function)
    print("to run Genetic Algorithm")
    fit, pt = mh.run(verbose=to_verbose, iterations=100, population=30, mutation_rate=0.1, crossover_rate=0.8, seed=115)
    print(fit)
    print(pt)
    if problem_to_solve == "partition sum":
        partition_problem.print_difference_subsets(pt)
    if problem_to_solve == "subset sum":
        subset_problem.print_difference_subsets(pt)

if "HS" in to_use:
    print("create Harmony Search")
    if problem_to_solve == "partition sum":
        mh = HSMH_Real(partition_problem.min_x, partition_problem.max_x, partition_problem.ndim, False, partition_problem.objective_function, partition_problem.repair_function, partition_problem.preprocess_function)
    if problem_to_solve == "subset sum":
        mh = HSMH_Real(subset_problem.min_x, subset_problem.max_x, subset_problem.ndim, False, subset_problem.objective_function, subset_problem.repair_function, subset_problem.preprocess_function)
    print("to run Harmony Search")
    fit, pt = mh.run(verbose=to_verbose, iterations=100, population=30, hmcr=0.9, par=0.3, bw=0.2, seed=115)
    print(fit)
    print(pt)
    if problem_to_solve == "partition sum":
        partition_problem.print_difference_subsets(pt)
    if problem_to_solve == "subset sum":
        subset_problem.print_difference_subsets(pt)
