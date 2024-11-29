#This file test the Real (Numerical) version of the Metaheuristics with the problems in anmetal.problems.nonlinear_functions using random vector data in a non linear distribution
import anmetal.problems.nonlinear_functions.n_inputs as problems_n
import anmetal.problems.nonlinear_functions.one_input as problems_1
import anmetal.problems.nonlinear_functions.two_inputs as problems_2

import numpy as np

#random values

def make_random_vector(cls_type, n):
    limits = cls_type.get_limits()
    vect = []
    if type(limits[0]) == type(list()):
        for i in range(len(limits)):
            vect.append(np.random.uniform(limits[i][0], limits[i][1]))
    elif type(limits[0]) == type(float()) or type(limits[0]) == type(int()):
        for i in range(n):
            vect.append(np.random.uniform(limits[0], limits[1]))
    else:
        raise ValueError("limits are no valid")
    return vect

cls_ns = [problems_n.Sumsquares, problems_n.Sphere, problems_n.Schwefel,
problems_n.Rosenbrock, problems_n.Rastrigrin, problems_n.Quartic, problems_n.Penalty,
#problems_n.Michalewicz, #the theorical min is not fixed, depends on dimension
problems_n.Griewank,
#below from first paper
problems_n.F15n, problems_n.F10n, problems_n.Brown3, problems_n.Brown1]


cls_1s = [problems_1.F1, problems_1.F3]
cls_2s = [problems_2.Camelback, problems_2.Goldsteinprice]

def eval_random(cls_list, n):
    rs = {}
    for cls_type in cls_list:
        vect = make_random_vector(cls_type, n)
        val = cls_type.func(vect)
        print(str(cls_type))
        print("value obtained: ", val, " and theoretical min: ", cls_type.get_theoretical_optimum())
        print("difference: ", abs(cls_type.get_theoretical_optimum()-val))
        rs[str(cls_type)] = {"vect": vect, "val": val, "theory": cls_type.get_theoretical_optimum()}
        if val < cls_type.get_theoretical_optimum():
            print("!"*40)
            print("WTF, it's less than minimum, something wrong")
    return rs

#random points
if False:
    print("#"*80)
    print("ns with 20 length vector")
    rn = eval_random(cls_ns, 30)
    print("#"*80)
    print("1s")
    r1 = eval_random(cls_1s, 1)
    print("#"*80)
    print("2s")
    r2 = eval_random(cls_2s, 2)


from anmetal.population.AFSA.AFSAMH_Real import AFSAMH_Real
from anmetal.population.SillyRandom.GreedyMH_Real import GreedyMH_Real
from anmetal.population.SillyRandom.GreedyMH_Real_WithLeap import GreedyMH_Real_WithLeap
from anmetal.population.PSO.PSOMH_Real import PSOMH_Real
from anmetal.population.PSO.PSOMH_Real_WithLeap import PSOMH_Real_WithLeap

def eval_mh(cls_list, n, mh_to_use, verbose=False):
    rs = {}
    for cls_type in cls_list:
        limits = cls_type.get_limits()
        if type(limits[0]) == type(float()) or type(limits[0]) == type(int()):
            min_x = limits[0]
            max_x = limits[1]
        else:
            print("skipping ", str(cls_type))
            continue
        if mh_to_use == "AFSA":
            mh = AFSAMH_Real(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, visual_distance_percentage=0.5, velocity_percentage=0.5, n_points_to_choose=3, crowded_percentage=0.7, its_stagnation=4, leap_percentage=0.3, stagnation_variation=0.4, seed=115)
        if mh_to_use == "Greedy":
            mh = GreedyMH_Real(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, iterations=100, population=30, seed=115)
        if mh_to_use == "GreedyWL":
            mh = GreedyMH_Real_WithLeap(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, iterations=100, population=30, stagnation_variation=0.4, its_stagnation=5, leap_percentage=0.8, seed=115)
        if mh_to_use == "PSO":
            mh = PSOMH_Real(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, iterations=100, population=30, omega=0.8, phi_g=0.5, phi_p=0.5 ,seed=115)
        if mh_to_use == "PSOWL":
            mh = PSOMH_Real_WithLeap(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, iterations=100, population=30, omega=0.8, phi_g=0.5, phi_p=0.5 ,seed=115, stagnation_variation=0.4, its_stagnation=5, leap_percentage=0.8)
        #vect = make_random_vector(cls_type, n)
        #val = cls_type.func(vect)
        print(str(cls_type))
        print("value obtained: ", fit, " and theoretical min: ", cls_type.get_theoretical_optimum())
        print("difference: ", abs(cls_type.get_theoretical_optimum()-fit))
        rs[str(cls_type)] = {"vec": pt, "val": fit, "theory": cls_type.get_theoretical_optimum(),
        "diff": abs(cls_type.get_theoretical_optimum()-fit)}
        if fit < cls_type.get_theoretical_optimum():
            print("!"*40)
            print("WTF, it's less than minimum, something wrong")
    return rs

if True:
    print("#"*80)
    print("ns with 30 length vector")
    mh_to_use = "AFSA"  # can be AFSA, Greedy, GreedyWL, PSO, PSOWL
    mh_rn = eval_mh(cls_ns, 30, mh_to_use)
    print("#"*80)
    print("1s")  # problems with 1D vector
    mh_r1 = eval_mh(cls_1s, 1, mh_to_use)
    print("#"*80)
    print("2s")  # problems with 2D vector
    mh_r2 = eval_mh(cls_2s, 2, mh_to_use)
