import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import argparse
import shutil
import os
from os.path import exists, join
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.switch_backend('Agg')
import cv2
import glob
import csv

from anmetal.optimizer.population.Genetic.GeneticMH_Categorical import GeneticMH_Categorical
from anmetal.optimizer.population.Genetic.GeneticMH_Categorical_WithLeap import GeneticMH_Categorical_WithLeap
from anmetal.problems.nphard_categorical.knapsack import Knapsack_Categorical
from anmetal.problems.nphard_categorical.sudoku import Sudoku
from anmetal.problems.nphard_categorical.sudoku_optimized import SudokuOptimized

def create_video_from_images(image_folder, output_video, fps=10):
    """Create MP4 video from sequence of PNG images"""
    images = sorted(glob.glob(os.path.join(image_folder, "gen_cat_*.png")), 
                   key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))
    
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

def main():
    parser = argparse.ArgumentParser(description='Genetic Algorithm Categorical Visualization')
    
    # Output Parameters
    parser.add_argument('--folder', type=str, default='mh_graphs',
                        help='Output folder for plots and videos (default: mh_graphs)')
    
    # GA Parameters
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--iterations", default=50, type=int, help="Number of iterations")
    parser.add_argument("--population", default=20, type=int, help="Population size")
    parser.add_argument("--elitist", default=0.3, type=float, help="Elitist percentage")
    parser.add_argument("--mutability", default=0.1, type=float, help="Mutability")
    parser.add_argument("--fidelity", default=1, type=int, help="Fidelity (1 for True, 0 for False)")
    parser.add_argument("--parents_mutation", default=1, type=int, help="Mutation in parents (1 for True, 0 for False)")
    parser.add_argument("--leap", default=0, type=int, help="Use Leap variant (1 for True, 0 for False)")
    
    # Visualization Parameters
    parser.add_argument("--categorytype", default="character", choices=["character", "color", "icon", "number", "value", "colorvalue"], help="Visualization type for categories")
    parser.add_argument("--fps", default=5, type=int, help="Frames per second for video")
    
    # Problem Parameters
    parser.add_argument("--problem", default="knapsack", choices=["knapsack", "sudoku", "sudoku_opt"], help="Problem to solve")
    parser.add_argument("--dims", default=20, type=int, help="Number of dimensions (for Knapsack)")
    
    args = parser.parse_args()
    print("Args:", args)

    seed = args.seed
    random_generator = RandomState(seed)
    
    # Setup Problem
    if args.problem == "knapsack":
        # Knapsack setup
        capacity = int(args.dims * 0.5 * 10) # Heuristic capacity
        problem = Knapsack_Categorical(knapsack_capacity=capacity, total_posible_elements=args.dims, seed=seed, max_cost=20, max_value=20)
        categorics = problem.get_possible_categories()
        ndims = args.dims
        to_max = True
        
        def obj_func(point):
            val = problem.objective_function(point)
            return val if val is not False else 0 # Handle invalid
            
        repair_func = problem.repair_function
        preprocess_func = problem.preprocess_function
        
    elif args.problem == "sudoku":
        # Sudoku setup
        problem = Sudoku()
        ndims = 81
        categorics = [[1, 2, 3, 4, 5, 6, 7, 8, 9] for _ in range(81)]
        to_max = False # Minimize violations
        
        def obj_func(point):
            # Convert 1D to 9x9 for validation
            # Sudoku.get_violations expects self.state to be set or we can modify it to take state
            # But get_violations uses self.state.
            # We need to be careful not to mess up shared state if parallel, but here it's sequential.
            old_state = problem.state
            problem.state = np.array(point).reshape((9, 9))
            violations = problem.get_violations()
            problem.state = old_state
            return violations

        def repair_func(point):
            return point # No repair implemented for Sudoku in this context
            
        def preprocess_func(point):
            return point

    elif args.problem == "sudoku_opt":
        # Sudoku Optimized setup
        problem = SudokuOptimized()
        ndims = 81
        categorics = problem.get_categories()
        to_max = False # Minimize violations
        
        def obj_func(point):
            return problem.objective_function(point)

        def repair_func(point):
            return problem.repair_function(point)
            
        def preprocess_func(point):
            return problem.preprocess_function(point)

    # Setup Algorithm
    if args.leap:
        mh = GeneticMH_Categorical_WithLeap(categorics, ndims, to_max, obj_func, repair_func, preprocess_func)
        gen = mh.run_yielded(iterations=args.iterations, population=args.population, 
                             elitist_percentage=args.elitist, mutability=args.mutability, 
                             fidelity=bool(args.fidelity), mutation_in_parents=bool(args.parents_mutation), 
                             seed=seed, verbose=True)
    else:
        mh = GeneticMH_Categorical(categorics, ndims, to_max, obj_func, repair_func, preprocess_func)
        gen = mh.run_yielded(iterations=args.iterations, population=args.population, 
                             elitist_percentage=args.elitist, mutability=args.mutability, 
                             fidelity=bool(args.fidelity), mutation_in_parents=bool(args.parents_mutation), 
                             seed=seed, verbose=True)

    # Output setup
    folderpath = join(args.folder, f"genetic_{args.problem}_{args.categorytype}")
    if exists(folderpath):
        shutil.rmtree(folderpath, ignore_errors=True)
    os.makedirs(folderpath)
    print(f"Output folder: {folderpath}")
    
    csv_file = open(join(folderpath, "history.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Iteration", "BestFitness", "BestSolution"])

    # Color map for categories
    unique_cats = set()
    for cat_list in categorics:
        for c in cat_list:
            unique_cats.add(c)
    unique_cats = sorted(list(unique_cats), key=lambda x: str(x))
    
    # Generate colors for categories
    cat_colors = {}
    for i, cat in enumerate(unique_cats):
        cat_colors[cat] = plt.cm.tab20(i % 20)

    print("Starting optimization...")
    
    history = []
    
    for iteration, best_fitness, best_point, points, fitnesses in gen:
        print(f"Processing iteration {iteration}...", end='\r')
        
        # Save to CSV
        csv_writer.writerow([iteration, best_fitness, str(best_point)])
        
        # Visualization
        fig = plt.figure(figsize=(15, max(8, args.population * 0.5)))
        
        # Grid layout: Main area for population, Right side for fitness bars
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
        ax_pop = fig.add_subplot(gs[0])
        ax_fit = fig.add_subplot(gs[1])
        
        # Draw Population
        ax_pop.set_xlim(0, ndims)
        ax_pop.set_ylim(0, args.population)
        ax_pop.set_aspect('equal')
        ax_pop.axis('off')
        ax_pop.set_title(f"Population - Iteration {iteration}")
        
        for i, point in enumerate(points):
            # Row i (from top to bottom, so y = population - 1 - i)
            y = args.population - 1 - i
            
            for j, val in enumerate(point):
                x = j
                
                # Draw rectangle
                rect_color = 'white'
                if args.categorytype in ["color", "colorvalue"]:
                    rect_color = cat_colors.get(val, 'white')
                
                rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='black', facecolor=rect_color)
                ax_pop.add_patch(rect)
                
                # Draw content
                if args.categorytype in ["character", "number", "icon", "value", "colorvalue"]:
                    text_val = str(val)
                    if args.categorytype == "character":
                        text_val = str(val)[0]
                    elif args.categorytype == "icon":
                        # Simple mapping for icon-like characters
                        if val == "is": text_val = "●"
                        elif val == "not": text_val = "○"
                        else: text_val = str(val)[0]
                        
                    ax_pop.text(x + 0.5, y + 0.5, text_val, 
                                horizontalalignment='center', 
                                verticalalignment='center', 
                                fontsize=8)

        # Draw Fitness
        y_pos = np.arange(len(fitnesses))
        # Reverse y_pos to match population order (top is index 0)
        y_pos = np.arange(len(fitnesses))[::-1]
        
        ax_fit.barh(y_pos, fitnesses, align='center')
        ax_fit.set_yticks(y_pos)
        ax_fit.set_yticklabels([f"Sol {i}" for i in range(len(fitnesses))])
        ax_fit.set_xlabel('Fitness')
        ax_fit.set_title('Fitness Values')
        
        # Add value labels
        for i, v in enumerate(fitnesses):
            ax_fit.text(v, y_pos[i], f"{v:.2f}", va='center')

        plt.tight_layout()
        plt.savefig(join(folderpath, f"gen_cat_{iteration:04d}.png"), dpi=100)
        plt.close(fig)

    csv_file.close()
    print("\nOptimization finished.")
    
    # Create Video
    video_path = join(folderpath, f"genetic_{args.problem}_animation.mp4")
    create_video_from_images(folderpath, video_path, args.fps)

if __name__ == "__main__":
    main()
