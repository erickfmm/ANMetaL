#This file plots the trajectories of Real Metaheuristics with NP-Complete problems
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
import os

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

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Plot trajectories of Real Metaheuristics with NP-Complete problems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run PSO and ABC on partition sum problem
  python trajectory_plot_npcomp.py --algorithms PSO ABC --problem "partition sum" --max-iterations 100
  
  # Run AFSA with custom parameters
  python trajectory_plot_npcomp.py --algorithms AFSA --parameter-afsa-visualdistancepercentage 0.6 --parameter-afsa-leappercentage 0.4
  
  # Run multiple algorithms with custom output folder
  python trajectory_plot_npcomp.py --algorithms PSO ABC ACO --folder ./results --ndims 1000 --seed 42
        """
    )
    
    # Basic configuration
    parser.add_argument('--folder', type=str, default='.',
                        help='Output folder for plots (default: current directory)')
    parser.add_argument('--algorithms', nargs='+', 
                        choices=['AFSA', 'Greedy', 'GreedyWL', 'PSO', 'PSOWL', 'ABC', 'ACO', 'BAT', 'BH', 'CUCKOO', 'FIREFLY', 'HS'],
                        default=['PSO', 'ABC', 'ACO', 'BAT', 'FIREFLY', 'HS'],
                        help='Algorithms to run (default: PSO ABC ACO BAT FIREFLY HS)')
    parser.add_argument('--problem', type=str, choices=['partition sum', 'subset sum'],
                        default='partition sum',
                        help='Problem to solve (default: partition sum)')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0,
                        help='Verbose level: 0=quiet, 1=normal, 2=debug (default: 0)')
    parser.add_argument('--max-iterations', type=int, default=50,
                        help='Maximum number of iterations (default: 50)')
    parser.add_argument('--population-size', type=int, default=20,
                        help='Population size (default: 20)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for algorithms (optional, default: 115)')
    parser.add_argument('--ndims', type=int, default=500,
                        help='Number of dimensions for the problem (default: 500)')
    
    # Algorithm-specific parameters - AFSA
    parser.add_argument('--parameter-afsa-visualdistancepercentage', type=float, default=0.5,
                        help='AFSA: Visual distance percentage (default: 0.5)')
    parser.add_argument('--parameter-afsa-velocitypercentage', type=float, default=0.5,
                        help='AFSA: Velocity percentage (default: 0.5)')
    parser.add_argument('--parameter-afsa-npointstochoose', type=int, default=3,
                        help='AFSA: Number of points to choose (default: 3)')
    parser.add_argument('--parameter-afsa-crowdedpercentage', type=float, default=0.7,
                        help='AFSA: Crowded percentage (default: 0.7)')
    parser.add_argument('--parameter-afsa-itsstagnation', type=int, default=4,
                        help='AFSA: Iterations stagnation (default: 4)')
    parser.add_argument('--parameter-afsa-leappercentage', type=float, default=0.3,
                        help='AFSA: Leap percentage (default: 0.3)')
    parser.add_argument('--parameter-afsa-stagnationvariation', type=float, default=0.4,
                        help='AFSA: Stagnation variation (default: 0.4)')
    
    # Algorithm-specific parameters - GreedyWL
    parser.add_argument('--parameter-greedywl-stagnationvariation', type=float, default=0.4,
                        help='GreedyWL: Stagnation variation (default: 0.4)')
    parser.add_argument('--parameter-greedywl-itsstagnation', type=int, default=5,
                        help='GreedyWL: Iterations stagnation (default: 5)')
    parser.add_argument('--parameter-greedywl-leappercentage', type=float, default=0.8,
                        help='GreedyWL: Leap percentage (default: 0.8)')
    
    # Algorithm-specific parameters - PSO
    parser.add_argument('--parameter-pso-omega', type=float, default=0.8,
                        help='PSO: Omega (inertia weight) (default: 0.8)')
    parser.add_argument('--parameter-pso-phig', type=float, default=1.0,
                        help='PSO: Phi_g (global coefficient) (default: 1.0)')
    parser.add_argument('--parameter-pso-phip', type=float, default=0.5,
                        help='PSO: Phi_p (personal coefficient) (default: 0.5)')
    
    # Algorithm-specific parameters - PSOWL
    parser.add_argument('--parameter-psowl-omega', type=float, default=0.8,
                        help='PSOWL: Omega (inertia weight) (default: 0.8)')
    parser.add_argument('--parameter-psowl-phig', type=float, default=1.0,
                        help='PSOWL: Phi_g (global coefficient) (default: 1.0)')
    parser.add_argument('--parameter-psowl-phip', type=float, default=0.5,
                        help='PSOWL: Phi_p (personal coefficient) (default: 0.5)')
    parser.add_argument('--parameter-psowl-stagnationvariation', type=float, default=0.4,
                        help='PSOWL: Stagnation variation (default: 0.4)')
    parser.add_argument('--parameter-psowl-itsstagnation', type=int, default=5,
                        help='PSOWL: Iterations stagnation (default: 5)')
    parser.add_argument('--parameter-psowl-leappercentage', type=float, default=0.8,
                        help='PSOWL: Leap percentage (default: 0.8)')
    
    # Algorithm-specific parameters - ABC
    parser.add_argument('--parameter-abc-limit', type=int, default=20,
                        help='ABC: Limit parameter (default: 20)')
    
    # Algorithm-specific parameters - ACO
    parser.add_argument('--parameter-aco-evaporationrate', type=float, default=0.1,
                        help='ACO: Evaporation rate (default: 0.1)')
    parser.add_argument('--parameter-aco-alpha', type=float, default=1.0,
                        help='ACO: Alpha parameter (default: 1.0)')
    parser.add_argument('--parameter-aco-beta', type=float, default=2.0,
                        help='ACO: Beta parameter (default: 2.0)')
    
    # Algorithm-specific parameters - BAT
    parser.add_argument('--parameter-bat-fmin', type=float, default=0.0,
                        help='BAT: Minimum frequency (default: 0.0)')
    parser.add_argument('--parameter-bat-fmax', type=float, default=2.0,
                        help='BAT: Maximum frequency (default: 2.0)')
    parser.add_argument('--parameter-bat-a', type=float, default=0.9,
                        help='BAT: Loudness (default: 0.9)')
    parser.add_argument('--parameter-bat-r0', type=float, default=0.9,
                        help='BAT: Pulse rate (default: 0.9)')
    
    # Algorithm-specific parameters - CUCKOO
    parser.add_argument('--parameter-cuckoo-pa', type=float, default=0.25,
                        help='CUCKOO: Probability of abandonment (default: 0.25)')
    
    # Algorithm-specific parameters - FIREFLY
    parser.add_argument('--parameter-firefly-alpha', type=float, default=0.5,
                        help='FIREFLY: Alpha (randomness) (default: 0.5)')
    parser.add_argument('--parameter-firefly-beta0', type=float, default=1.0,
                        help='FIREFLY: Beta0 (attractiveness) (default: 1.0)')
    parser.add_argument('--parameter-firefly-gamma', type=float, default=1.0,
                        help='FIREFLY: Gamma (absorption coefficient) (default: 1.0)')
    
    # Algorithm-specific parameters - HS
    parser.add_argument('--parameter-hs-hmcr', type=float, default=0.9,
                        help='HS: Harmony Memory Considering Rate (default: 0.9)')
    parser.add_argument('--parameter-hs-par', type=float, default=0.3,
                        help='HS: Pitch Adjustment Rate (default: 0.3)')
    parser.add_argument('--parameter-hs-bw', type=float, default=0.2,
                        help='HS: Bandwidth (default: 0.2)')
    
    return parser.parse_args()

# Parse command-line arguments
args = parse_arguments()

# Configuration from arguments
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

# Select which algorithms to run from command line
algorithms_to_run = args.algorithms

# Create output folder if it doesn't exist
output_folder = args.folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Problem setup with command-line parameters
partition_problem = Partition_Real(seed=42, num_dims=args.ndims)
subset_problem = Subset_Real(seed=42, num_dims=args.ndims)

problem_to_solve = args.problem

to_verbose = args.verbose > 0
max_iterations = args.max_iterations
population_size = args.population_size
run_seed = args.seed if args.seed is not None else 115

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
            verbose=to_verbose, seed=run_seed, **kwargs):
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
                          AFSAMH_Real, 
                          visual_distance_percentage=args.parameter_afsa_visualdistancepercentage,
                          velocity_percentage=args.parameter_afsa_velocitypercentage,
                          n_points_to_choose=args.parameter_afsa_npointstochoose,
                          crowded_percentage=args.parameter_afsa_crowdedpercentage,
                          its_stagnation=args.parameter_afsa_itsstagnation,
                          leap_percentage=args.parameter_afsa_leappercentage,
                          stagnation_variation=args.parameter_afsa_stagnationvariation)

if "GreedyWL" in algorithms_to_run and "GreedyWL" in to_use:
    collect_trajectory_data("GreedyWL", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          GreedyMH_Real_WithLeap,
                          stagnation_variation=args.parameter_greedywl_stagnationvariation,
                          its_stagnation=args.parameter_greedywl_itsstagnation,
                          leap_percentage=args.parameter_greedywl_leappercentage)

if "Greedy" in algorithms_to_run and "Greedy" in to_use:
    collect_trajectory_data("Greedy", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          GreedyMH_Real)

if "PSO" in algorithms_to_run and "PSO" in to_use:
    collect_trajectory_data("PSO", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          PSOMH_Real,
                          omega=args.parameter_pso_omega,
                          phi_g=args.parameter_pso_phig,
                          phi_p=args.parameter_pso_phip)

if "PSOWL" in algorithms_to_run and "PSOWL" in to_use:
    collect_trajectory_data("PSOWL", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          PSOMH_Real_WithLeap,
                          omega=args.parameter_psowl_omega,
                          phi_g=args.parameter_psowl_phig,
                          phi_p=args.parameter_psowl_phip,
                          stagnation_variation=args.parameter_psowl_stagnationvariation,
                          its_stagnation=args.parameter_psowl_itsstagnation,
                          leap_percentage=args.parameter_psowl_leappercentage)

if "ABC" in algorithms_to_run and "ABC" in to_use:
    collect_trajectory_data("ABC", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          ABCMH_Real,
                          limit=args.parameter_abc_limit)

if "ACO" in algorithms_to_run and "ACO" in to_use:
    collect_trajectory_data("ACO", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          ACOMH_Real,
                          evaporation_rate=args.parameter_aco_evaporationrate,
                          alpha=args.parameter_aco_alpha,
                          beta=args.parameter_aco_beta)

if "BAT" in algorithms_to_run and "BAT" in to_use:
    collect_trajectory_data("BAT", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          BatMH_Real,
                          fmin=args.parameter_bat_fmin,
                          fmax=args.parameter_bat_fmax,
                          A=args.parameter_bat_a,
                          r0=args.parameter_bat_r0)

if "BH" in algorithms_to_run and "BH" in to_use:
    collect_trajectory_data("Blackhole", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          BlackholeMH_Real)

if "CUCKOO" in algorithms_to_run and "CUCKOO" in to_use:
    collect_trajectory_data("Cuckoo", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          CuckooMH_Real,
                          pa=args.parameter_cuckoo_pa)

if "FIREFLY" in algorithms_to_run and "FIREFLY" in to_use:
    collect_trajectory_data("Firefly", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          FireflyMH_Real,
                          alpha=args.parameter_firefly_alpha,
                          beta0=args.parameter_firefly_beta0,
                          gamma=args.parameter_firefly_gamma)

if "HS" in algorithms_to_run and "HS" in to_use:
    collect_trajectory_data("Harmony Search", partition_problem if problem_to_solve == "partition sum" else subset_problem,
                          HSMH_Real,
                          hmcr=args.parameter_hs_hmcr,
                          par=args.parameter_hs_par,
                          bw=args.parameter_hs_bw)

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
    output_path = os.path.join(output_folder, f'metaheuristics_trajectories_{problem_to_solve.replace(" ", "_")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved trajectory plot to: {output_path}")
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
    output_path = os.path.join(output_folder, f'metaheuristics_seaborn_{problem_to_solve.replace(" ", "_")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved seaborn plot to: {output_path}")
    plt.show()
    
else:
    print("No trajectory data collected. Please check the algorithm implementations.")
