#This file test the Real version of the Metaheuristics (Numerical) with the problems in anmetal.problems.nphard_real package that works on 2D vector points, creating a figure in each iteration with the position of each point and the position of the just past iteration with a line between them for each point. to view the evolution and movement of the solution candidates (points) in the search space

# Import all population metaheuristics (except genetic algorithm)
from anmetal.population.AFSA.AFSAMH_Real import AFSAMH_Real
from anmetal.population.SillyRandom.GreedyMH_Real import GreedyMH_Real
from anmetal.population.SillyRandom.GreedyMH_Real_WithLeap import GreedyMH_Real_WithLeap
from anmetal.population.PSO.PSOMH_Real import PSOMH_Real
from anmetal.population.PSO.PSOMH_Real_WithLeap import PSOMH_Real_WithLeap
from anmetal.population.ABC.abc import ArtificialBeeColony
from anmetal.population.AntColony.aco import AntColony
from anmetal.population.Bat.bat import BatAlgorithm
from anmetal.population.Blackhole.blackhole import BlackHole
from anmetal.population.Cuckoo.cuckoo import CuckooSearch
from anmetal.population.Firefly.firefly import FireflyAlgorithm
from anmetal.population.HarmonySearch.harmony import HarmonySearch


#from problems.nphard_real.partition__and_subset_sum import Partition_Real, Subset_Real
import anmetal.problems.nonlinear_functions.two_inputs as problems_2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import RandomState
import numpy as np
import os
from os.path import exists, join
import cv2
import glob

import argparse
import shutil

parser = argparse.ArgumentParser(description='Metaheuristic Animation Generator - Creates visualizations and videos of metaheuristic optimization algorithms')
#parser.add_argument("--a", default=1, type=int, help="This is the 'a' variable")

parser.add_argument("--seed", default=0, type=int, help="The integer to be seed of random number generator")
parser.add_argument("--mh", default="", type=str, help="\
    Name of Metaheuristic, can be one of:\n\
    AFSA\n\
    PSO\n\
    PSOWL\n\
    Greed\n\
    GreedWL\n\
    ABC\n\
    ACO\n\
    Bat\n\
    Blackhole\n\
    Cuckoo\n\
    Firefly\n\
    Harmony")
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
parser.add_argument("--plot3d", default=0, type=int, help="1 for 3D plot with fitness as Z-axis, 0 for 2D plot")
parser.add_argument("--fps", default=10, type=int, help="Frames per second for the output video")

args = parser.parse_args()

print("args: ", args)

to_verbose : bool = True if args.verbose == 1 else False
plot_3d : bool = True if args.plot3d == 1 else False
seed: int = args.seed
random_generator = RandomState(seed)
iterations = args.iterations
population = args.population
fps = args.fps

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

# Get problem limits properly - handle both formats
limits = prob.get_limits()
if isinstance(limits[0], list):
    # Format: ([x_min, x_max], [y_min, y_max])
    x_limits, y_limits = limits
    min_val = min(x_limits[0], y_limits[0])  # Use the most restrictive bounds
    max_val = max(x_limits[1], y_limits[1])
else:
    # Format: [min, max] (same for both x and y)
    min_val, max_val = limits
    x_limits = y_limits = limits

# Initialize metaheuristics based on user selection
mh_name = str.lower(args.mh)

if mh_name == "afsa" or args.mh == "":
    mh = AFSAMH_Real(min_val, max_val, 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, visual_distance_percentage=0.2, velocity_percentage=0.3, n_points_to_choose=5, crowded_percentage=0.8, its_stagnation=7, leap_percentage=0.2, stagnation_variation=0.4, seed=seed)
elif mh_name == "pso":
    mh = PSOMH_Real(min_val, max_val, 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, omega=0.5, phi_g=1, phi_p=2)
elif mh_name == "psowl":
    mh = PSOMH_Real_WithLeap(min_val, max_val, 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, omega=0.5, phi_g=1, phi_p=2, stagnation_variation=0.4, its_stagnation=5, leap_percentage=0.8)
elif mh_name == "greed":
    mh = GreedyMH_Real(min_val, max_val, 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population)
elif mh_name == "greedwl":
    mh = GreedyMH_Real_WithLeap(min_val, max_val, 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, stagnation_variation=0.4, its_stagnation=5, leap_percentage=0.8)
elif mh_name == "abc":
    mh = ArtificialBeeColony(min_val, max_val, 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, limit=20, seed=seed)
elif mh_name == "aco":
    mh = AntColony(min_val, max_val, 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, evaporation_rate=0.1, alpha=1.0, beta=2.0, seed=seed)
elif mh_name == "bat":
    mh = BatAlgorithm(min_val, max_val, 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, fmin=0, fmax=2, A=0.9, r0=0.9, seed=seed)
elif mh_name == "blackhole":
    mh = BlackHole(min_val, max_val, 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, seed=seed)
elif mh_name == "cuckoo":
    mh = CuckooSearch(min_val, max_val, 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, pa=0.25, seed=seed)
elif mh_name == "firefly":
    mh = FireflyAlgorithm(min_val, max_val, 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, alpha=0.5, beta0=1.0, gamma=1.0, seed=seed)
elif mh_name == "harmony":
    mh = HarmonySearch(min_val, max_val, 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, hmcr=0.9, par=0.3, bw=0.2, seed=seed)
else:
    print(f"Unknown metaheuristic: {args.mh}. Using AFSA as default.")
    mh = AFSAMH_Real(min_val, max_val, 2, False, prob.func, None, None)
    gen = mh.run_yielded(verbose=to_verbose, iterations=iterations, population=population, visual_distance_percentage=0.2, velocity_percentage=0.3, n_points_to_choose=5, crowded_percentage=0.8, its_stagnation=7, leap_percentage=0.2, stagnation_variation=0.4, seed=seed)

folderpath = join("mh_graphs", args.mh+"_"+args.problem)
if exists(folderpath):
    shutil.rmtree(folderpath, ignore_errors=True)
os.makedirs(folderpath)

def create_video_from_images(image_folder, output_video, fps=10):
    """Create MP4 video from sequence of PNG images"""
    images = sorted(glob.glob(os.path.join(image_folder, "mhgraph_*.png")), 
                   key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    if not images:
        print("No images found to create video")
        return
    
    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for image in images:
        frame = cv2.imread(image)
        video.write(frame)
    
    video.release()
    print(f"Video saved as: {output_video}")

print("to start iterations")
colors_to_use = []
is_first : bool = True

# Configure plot for 2D or 3D
if plot_3d:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
else:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
for iteration, best_fitness_historical, best_bin_point, points_a, fts in gen:
    print("iteration: ", iteration)
    
    # Clear previous plot
    ax.clear()
    
    # Set limits and labels
    if plot_3d:
        ax.set_xlim(x_limits[0], x_limits[1])
        ax.set_ylim(y_limits[0], y_limits[1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Fitness')
        ax.set_title(f'{args.mh.upper()} - {args.problem} - Iteration {iteration}')
    else:
        ax.set_xlim(x_limits[0], x_limits[1])
        ax.set_ylim(y_limits[0], y_limits[1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{args.mh.upper()} - {args.problem} - Iteration {iteration}')
    
    ps_ = np.copy(points_a)
    
    if is_first: # Generate colors for each point
        for i in range(len(ps_)):
            color = (random_generator.uniform(0.3, 1),
                    random_generator.uniform(0.3, 1),
                    random_generator.uniform(0.3, 1))
            colors_to_use.append(color)
            
            if plot_3d:
                fitness = fts[i] if fts is not None else prob.func(ps_[i])
                ax.scatter(ps_[i][0], ps_[i][1], fitness, c=[color], s=60, marker='*')
            else:
                ax.plot(ps_[i][0], ps_[i][1], '*', color=color, markersize=8)
                
        last_points = np.copy(ps_)
        last_fitness = np.copy(fts) if fts is not None else [prob.func(p) for p in ps_]
        is_first = False
    else:
        # Draw lines and points for movement visualization
        for i in range(len(ps_)):
            current_fitness = fts[i] if fts is not None else prob.func(ps_[i])
            
            if plot_3d:
                # Draw line from previous to current position
                ax.plot([last_points[i][0], ps_[i][0]], 
                       [last_points[i][1], ps_[i][1]], 
                       [last_fitness[i], current_fitness], 
                       '-', color=colors_to_use[i], alpha=0.7)
                # Draw current point
                ax.scatter(ps_[i][0], ps_[i][1], current_fitness, c=[colors_to_use[i]], s=60, marker='*')
            else:
                # Draw line from previous to current position
                ax.plot([last_points[i][0], ps_[i][0]], [last_points[i][1], ps_[i][1]], 
                       '-', color=colors_to_use[i], alpha=0.7)
                # Draw current point
                ax.plot(ps_[i][0], ps_[i][1], '*', color=colors_to_use[i], markersize=8)
        
        last_points = np.copy(ps_)
        last_fitness = [fts[i] if fts is not None else prob.func(ps_[i]) for i in range(len(ps_))]
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(join(folderpath, f"mhgraph_{iteration:04d}.png"), dpi=150, bbox_inches='tight')

# Create video from images
video_path = join(folderpath, f"{args.mh}_{args.problem}_animation.mp4")
create_video_from_images(folderpath, video_path, fps)

print(f"Animation completed! Images saved in: {folderpath}")
print(f"Video saved as: {video_path}")

# Usage examples:
# 2D plotting: python mh_graph_each_it.py --mh ABC --problem Camelback --iterations 20 --population 30
# 3D plotting: python mh_graph_each_it.py --mh Firefly --problem Goldsteinprice --iterations 15 --plot3d 1 --fps 30
# Available algorithms: AFSA, PSO, PSOWL, Greed, GreedWL, ABC, ACO, Bat, Blackhole, Cuckoo, Firefly, Harmony
# Available problems: Camelback, Goldsteinprice, Pshubert1, Pshubert2, Shubert, Quartic