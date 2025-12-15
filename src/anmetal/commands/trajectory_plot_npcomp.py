#This file plots the trajectories of Real Metaheuristics with NP-Complete problems
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from anmetal.optimizer.population.AFSA.AFSAMH_Real import AFSAMH_Real
from anmetal.optimizer.population.Greedy.GreedyMH_Real import GreedyMH_Real
from anmetal.optimizer.population.Greedy.GreedyMH_Real_WithLeap import GreedyMH_Real_WithLeap
from anmetal.optimizer.population.PSO.PSOMH_Real import PSOMH_Real
from anmetal.optimizer.population.PSO.PSOMH_Real_WithLeap import PSOMH_Real_WithLeap
from anmetal.optimizer.population.ABC.abc import ArtificialBeeColony as ABCMH_Real
from anmetal.optimizer.population.ACO.aco import AntColony as ACOMH_Real
from anmetal.optimizer.population.Bat.bat import BatAlgorithm as BatMH_Real
from anmetal.optimizer.population.Blackhole.blackhole import BlackHole as BlackholeMH_Real
from anmetal.optimizer.population.Cuckoo.cuckoo import CuckooSearch as CuckooMH_Real
from anmetal.optimizer.population.Firefly.firefly import FireflyAlgorithm as FireflyMH_Real
from anmetal.optimizer.population.Harmony.harmony import HarmonySearch as HSMH_Real

from anmetal.problems.nphard_real.partition__and_subset_sum import Partition_Real, Subset_Real

# Configuration
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
    "HS"
]

# Select which algorithms to run (reduce for faster execution)
algorithms_to_run = ["PSO", "ABC", "ACO", "BAT", "FIREFLY", "HS"]  # You can modify this list

partition_problem = Partition_Real(seed=42, num_dims=500)  # Reduced dimensions for faster execution
subset_problem = Subset_Real(seed=42, num_dims=500)

problem_to_solve = "partition sum"
#problem_to_solve = "subset sum"

to_verbose = False  # Set to False to avoid cluttering output
max_iterations = 50  # Reduced iterations for visualization
population_size = 20  # Reduced population for faster execution

# Storage for trajectory data
trajectory_data = []

def collect_trajectory_data(algorithm_name, problem, mh_class, **kwargs):
    """Collect trajectory data from a metaheuristic run"""
    print(f"Running {algorithm_name}...")
    
    if problem_to_solve == "partition sum":
        mh = mh_class(problem.min_x, problem.max_x, problem.ndim, False, 
                     problem.objective_function, problem.repair_function, 
                     problem.preprocess_function)
    else:
        mh = mh_class(problem.min_x, problem.max_x, problem.ndim, False, 
                     problem.objective_function, problem.repair_function, 
                     problem.preprocess_function)
    
    # Use run_yielded to get trajectory data
    iterations = []
    fitnesses = []
    
    try:
        for iteration, best_fitness, best_point, points, fts in mh.run_yielded(
            iterations=max_iterations, population=population_size, 
            verbose=to_verbose, seed=115, **kwargs):
            iterations.append(iteration)
            fitnesses.append(best_fitness)
            
            # Store data for plotting
            trajectory_data.append({
                'Algorithm': algorithm_name,
                'Iteration': iteration,
                'Best_Fitness': best_fitness,
                'Problem': problem_to_solve
            })
    except Exception as e:
        print(f"Error running {algorithm_name}: {e}")
        return None, None
    
    print(f"Final fitness for {algorithm_name}: {fitnesses[-1] if fitnesses else 'N/A'}")
    return iterations, fitnesses

# Run algorithms and collect data
if "AFSA" in algorithms_to_run and "AFSA" in to_use:
    collect_trajectory_data("AFSA", partition_problem if problem_to_solve == "partition sum" else subset_problem, 
                          AFSAMH_Real, visual_distance_percentage=0.5, velocity_percentage=0.5, 
                          n_points_to_choose=3, crowded_percentage=0.7, its_stagnation=4, 
                          leap_percentage=0.3, stagnation_variation=0.4)

if "GreedyWL" in algorithms_to_run and "GreedyWL" in to_use:
    collect_trajectory_data("GreedyWL", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          GreedyMH_Real_WithLeap, stagnation_variation=0.4, its_stagnation=5, leap_percentage=0.8)

if "Greedy" in algorithms_to_run and "Greedy" in to_use:
    collect_trajectory_data("Greedy", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          GreedyMH_Real)

if "PSO" in algorithms_to_run and "PSO" in to_use:
    collect_trajectory_data("PSO", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          PSOMH_Real, omega=0.8, phi_g=1, phi_p=0.5)

if "PSOWL" in algorithms_to_run and "PSOWL" in to_use:
    collect_trajectory_data("PSOWL", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          PSOMH_Real_WithLeap, omega=0.8, phi_g=1, phi_p=0.5, 
                          stagnation_variation=0.4, its_stagnation=5, leap_percentage=0.8)

if "ABC" in algorithms_to_run and "ABC" in to_use:
    collect_trajectory_data("ABC", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          ABCMH_Real, limit=20)

if "ACO" in algorithms_to_run and "ACO" in to_use:
    collect_trajectory_data("ACO", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          ACOMH_Real, evaporation_rate=0.1, alpha=1.0, beta=2.0)

if "BAT" in algorithms_to_run and "BAT" in to_use:
    collect_trajectory_data("BAT", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          BatMH_Real, fmin=0, fmax=2, A=0.9, r0=0.9)

if "BH" in algorithms_to_run and "BH" in to_use:
    collect_trajectory_data("Blackhole", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          BlackholeMH_Real)

if "CUCKOO" in algorithms_to_run and "CUCKOO" in to_use:
    collect_trajectory_data("Cuckoo", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          CuckooMH_Real, pa=0.25)

if "FIREFLY" in algorithms_to_run and "FIREFLY" in to_use:
    collect_trajectory_data("Firefly", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          FireflyMH_Real, alpha=0.5, beta0=1.0, gamma=1.0)

if "HS" in algorithms_to_run and "HS" in to_use:
    collect_trajectory_data("Harmony Search", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          HSMH_Real, hmcr=0.9, par=0.3, bw=0.2)

# Create DataFrame from collected data
df = pd.DataFrame(trajectory_data)

if len(df) > 0:
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create the main trajectory plot
    plt.figure(figsize=(14, 10))
    
    # Main plot: Fitness trajectories
    plt.subplot(2, 2, (1, 2))
    for algorithm in df['Algorithm'].unique():
        alg_data = df[df['Algorithm'] == algorithm]
        plt.plot(alg_data['Iteration'], alg_data['Best_Fitness'], 
                marker='o', markersize=3, linewidth=2, label=algorithm, alpha=0.8)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title(f'Metaheuristic Convergence Trajectories - {problem_to_solve.title()}', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Semi-log plot for better visualization of convergence
    plt.subplot(2, 2, 3)
    for algorithm in df['Algorithm'].unique():
        alg_data = df[df['Algorithm'] == algorithm]
        plt.semilogy(alg_data['Iteration'], alg_data['Best_Fitness'], 
                    marker='o', markersize=3, linewidth=2, label=algorithm, alpha=0.8)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness (log scale)', fontsize=12)
    plt.title('Semi-log Convergence View', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Final fitness comparison
    plt.subplot(2, 2, 4)
    final_fitness = df.groupby('Algorithm')['Best_Fitness'].last().reset_index()
    final_fitness = final_fitness.sort_values('Best_Fitness')
    
    bars = plt.bar(range(len(final_fitness)), final_fitness['Best_Fitness'], 
                   color=sns.color_palette("husl", len(final_fitness)))
    plt.xticks(range(len(final_fitness)), final_fitness['Algorithm'], rotation=45, ha='right')
    plt.ylabel('Final Best Fitness', fontsize=12)
    plt.title('Final Performance Comparison', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2e}' if height < 0.01 else f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'metaheuristics_trajectories_{problem_to_solve.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAJECTORY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Problem: {problem_to_solve.title()}")
    print(f"Dimensions: {partition_problem.ndim if problem_to_solve == 'partition sum' else subset_problem.ndim}")
    print(f"Iterations: {max_iterations}")
    print(f"Population: {population_size}")
    print("\nFinal Results:")
    print("-" * 40)
    
    final_results = df.groupby('Algorithm').agg({
        'Best_Fitness': ['first', 'last', 'min'],
        'Iteration': 'max'
    }).round(6)
    
    final_results.columns = ['Initial_Fitness', 'Final_Fitness', 'Best_Fitness', 'Total_Iterations']
    final_results['Improvement'] = ((final_results['Initial_Fitness'] - final_results['Final_Fitness']) / 
                                   final_results['Initial_Fitness'] * 100).round(2)
    final_results = final_results.sort_values('Final_Fitness')
    
    print(final_results.to_string())
    
    # Additional seaborn plot for convergence comparison
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='Iteration', y='Best_Fitness', hue='Algorithm', 
                marker='o', markersize=4, linewidth=2)
    plt.title(f'Metaheuristic Convergence Comparison - {problem_to_solve.title()}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the seaborn plot
    plt.savefig(f'metaheuristics_seaborn_{problem_to_solve.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
else:
    print("No trajectory data collected. Please check the algorithm implementations.")
