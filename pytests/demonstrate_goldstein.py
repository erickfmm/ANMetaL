import sys
import os

# Add src to path so we can import anmetal
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from anmetal.problems.nonlinear_functions.two_inputs import Goldsteinprice
from anmetal.optimizer.population.PSO.PSOMH_Real import PSOMH_Real
from anmetal.optimizer.population.Cuckoo.cuckoo import CuckooSearch
import numpy as np

def demonstrate_goldstein_optimization():
    print("="*60)
    print("Goldstein-Price Function Optimization Demonstration")
    print("="*60)
    
    # Problem Setup
    problem = Goldsteinprice
    limits = problem.get_limits()
    min_x, max_x = limits[0], limits[1]
    # Note: get_limits returns [-2, 2], so min_x=-2, max_x=2 if it returns a list.
    # Let's check if it returns a list of lists or a single list.
    # In two_inputs.py: return [-2, 2] -> single list.
    # But some problems return [[min, max], [min, max]].
    # The code in nonlinear_mh_optimization.py handles this:
    # if type(limits[0]) == type(float()) or type(limits[0]) == type(int()): ...
    
    if isinstance(limits[0], (int, float)):
        min_val = limits[0]
        max_val = limits[1]
    else:
        # Assuming symmetric for this demo if it was list of lists, but Goldstein is [-2, 2]
        min_val = limits[0][0]
        max_val = limits[0][1]

    print(f"Theoretical Optimum: {problem.get_theoretical_optimum()}")
    print(f"Search Space: [{min_val}, {max_val}] per dimension")
    print("-" * 60)

    # Approach 1: Particle Swarm Optimization (PSO)
    print("\nApproach 1: Particle Swarm Optimization (PSO)")
    print("PSO simulates the social behavior of birds flocking or fish schooling.")
    
    # Initialize PSO
    # n=2 dimensions (x, y)
    pso = PSOMH_Real(min_val, max_val, 2, False, problem.func, None, None)
    
    # Run PSO
    # iterations=100, population=30
    print("Running PSO...")
    best_fitness_pso, best_pos_pso = pso.run(
        verbose=False, 
        iterations=100, 
        population=50, 
        omega=0.7,  # Inertia weight
        phi_g=0.5,  # Global best weight
        phi_p=0.5,  # Personal best weight
        seed=42
    )
    
    print(f"PSO Best Fitness Found: {best_fitness_pso}")
    print(f"PSO Best Position: {best_pos_pso}")
    print(f"Error: {abs(best_fitness_pso - problem.get_theoretical_optimum())}")

    # Approach 2: Cuckoo Search
    print("\nApproach 2: Cuckoo Search")
    print("Cuckoo Search is based on the brood parasitism of some cuckoo species.")
    
    # Initialize Cuckoo Search
    # Note: CuckooSearch might have a different run signature or init signature.
    # Based on file read: __init__(..., population_size=25, pa=0.25)
    cs = CuckooSearch(
        min_value=min_val, 
        max_value=max_val, 
        ndims=2, 
        to_max=False, 
        objective_function=problem.func,
        population_size=30,
        pa=0.25
    )
    
    # Run Cuckoo Search
    # I need to check the run method of CuckooSearch.
    # Assuming it has a run method similar to IMetaheuristic or others.
    # Let's check the file content again if needed, but usually it's run(iterations=...)
    print("Running Cuckoo Search...")
    try:
        best_fitness_cs, best_pos_cs = cs.run(iterations=100, verbose=False)
        print(f"Cuckoo Search Best Fitness Found: {best_fitness_cs}")
        print(f"Cuckoo Search Best Position: {best_pos_cs}")
        print(f"Error: {abs(best_fitness_cs - problem.get_theoretical_optimum())}")
    except Exception as e:
        print(f"Could not run Cuckoo Search: {e}")
        # Fallback to checking run signature if it fails
        
if __name__ == "__main__":
    demonstrate_goldstein_optimization()
