#This file test the Real (Numerical) version of the Metaheuristics with the problems in anmetal.problems.nonlinear_functions using random vector data in a non linear distribution
import anmetal.problems.nonlinear_functions.n_inputs as problems_n
import anmetal.problems.nonlinear_functions.one_input as problems_1
import anmetal.problems.nonlinear_functions.two_inputs as problems_2

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


from anmetal.optimizer.population.AFSA.AFSAMH_Real import AFSAMH_Real
from anmetal.optimizer.population.Greedy.GreedyMH_Real import GreedyMH_Real
from anmetal.optimizer.population.Greedy.GreedyMH_Real_WithLeap import GreedyMH_Real_WithLeap
from anmetal.optimizer.population.PSO.PSOMH_Real import PSOMH_Real
from anmetal.optimizer.population.PSO.PSOMH_Real_WithLeap import PSOMH_Real_WithLeap
from anmetal.optimizer.population.ABC.abc import ArtificialBeeColony
from anmetal.optimizer.population.ACO.aco import AntColony
from anmetal.optimizer.population.Bat.bat import BatAlgorithm
from anmetal.optimizer.population.Blackhole.blackhole import BlackHole
from anmetal.optimizer.population.Cuckoo.cuckoo import CuckooSearch
from anmetal.optimizer.population.Firefly.firefly import FireflyAlgorithm
from anmetal.optimizer.population.Harmony.harmony import HarmonySearch

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
        elif mh_to_use == "Greedy":
            mh = GreedyMH_Real(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, iterations=100, population=30, seed=115)
        elif mh_to_use == "GreedyWL":
            mh = GreedyMH_Real_WithLeap(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, iterations=100, population=30, stagnation_variation=0.4, its_stagnation=5, leap_percentage=0.8, seed=115)
        elif mh_to_use == "PSO":
            mh = PSOMH_Real(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, iterations=100, population=30, omega=0.8, phi_g=0.5, phi_p=0.5 ,seed=115)
        elif mh_to_use == "PSOWL":
            mh = PSOMH_Real_WithLeap(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, iterations=100, population=30, omega=0.8, phi_g=0.5, phi_p=0.5 ,seed=115, stagnation_variation=0.4, its_stagnation=5, leap_percentage=0.8)
        elif mh_to_use == "ABC":
            mh = ArtificialBeeColony(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, iterations=100, population=30, limit=20, seed=115)
        elif mh_to_use == "ACO":
            mh = AntColony(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, iterations=100, population=30, evaporation_rate=0.1, alpha=1.0, beta=2.0, seed=115)
        elif mh_to_use == "Bat":
            mh = BatAlgorithm(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, iterations=100, population=30, fmin=0, fmax=2, A=0.9, r0=0.9, seed=115)
        elif mh_to_use == "Blackhole":
            mh = BlackHole(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, iterations=100, population=30, seed=115)
        elif mh_to_use == "Cuckoo":
            mh = CuckooSearch(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, iterations=100, population=30, pa=0.25, seed=115)
        elif mh_to_use == "Firefly":
            mh = FireflyAlgorithm(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, iterations=100, population=30, alpha=0.5, beta0=1.0, gamma=1.0, seed=115)
        elif mh_to_use == "HarmonySearch":
            mh = HarmonySearch(min_x, max_x, n, False, cls_type.func, None, None)
            fit, pt = mh.run(verbose=verbose, iterations=100, population=30, hmcr=0.9, par=0.3, bw=0.2, seed=115)
        else:
            print(f"Unknown metaheuristic: {mh_to_use}")
            continue
        
        print(str(cls_type))
        print("value obtained: ", fit, " and theoretical min: ", cls_type.get_theoretical_optimum())
        print("difference: ", abs(cls_type.get_theoretical_optimum()-fit))
        rs[str(cls_type)] = {"vec": pt, "val": fit, "theory": cls_type.get_theoretical_optimum(),
        "diff": abs(cls_type.get_theoretical_optimum()-fit)}
        if fit < cls_type.get_theoretical_optimum():
            print("!"*40)
            print("WTF, it's less than minimum, something wrong")
    return rs

def eval_with_all_mh(cls_list, verbose=False):
    """
    Evaluate all metaheuristics (except genetic algorithms) with all problems in cls_list.
    
    Args:
        cls_list: List of problem classes to test
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with results for each metaheuristic and problem combination
    """
    # List of all population metaheuristics (excluding genetic algorithms)
    metaheuristics = [
        "AFSA", "Greedy", "GreedyWL", "PSO", "PSOWL", 
        "ABC", "ACO", "Bat", "Blackhole", "Cuckoo", 
        "Firefly", "HarmonySearch"
    ]
    
    all_results = {}
    
    for mh_name in metaheuristics:
        print("="*80)
        print(f"Testing {mh_name} metaheuristic")
        print("="*80)
        
        # Determine the appropriate dimension based on problem types
        # Check if this is from n_inputs, one_input, or two_inputs
        if any("n_inputs" in str(cls) for cls in cls_list):
            n_dims = 30  # For n-dimensional problems
        elif any("one_input" in str(cls) for cls in cls_list):
            n_dims = 1   # For 1D problems
        elif any("two_inputs" in str(cls) for cls in cls_list):
            n_dims = 2   # For 2D problems
        else:
            # Default dimension - try to infer from first problem
            try:
                limits = cls_list[0].get_limits()
                if isinstance(limits[0], list):
                    n_dims = len(limits)
                else:
                    n_dims = 30  # Default fallback
            except:
                n_dims = 30  # Default fallback
        
        try:
            mh_results = eval_mh(cls_list, n_dims, mh_name, verbose=verbose)
            all_results[mh_name] = mh_results
        except Exception as e:
            print(f"Error testing {mh_name}: {str(e)}")
            all_results[mh_name] = {"error": str(e)}
        
        print(f"Completed {mh_name}")
        print()
    
    return all_results

if False:  # Set to True to run the tests
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

# Example usage of eval_with_all_mh function
if True:  # Set to True to run comprehensive testing
    # Define metaheuristics list for plotting
    metaheuristics = [
        "AFSA", "Greedy", "GreedyWL", "PSO", "PSOWL", 
        "ABC", "ACO", "Bat", "Blackhole", "Cuckoo", 
        "Firefly", "HarmonySearch"
    ]
    
    print("#"*80)
    print("Testing all metaheuristics with n-dimensional problems")
    all_mh_results_n = eval_with_all_mh(cls_ns, verbose=False)
    
    print("#"*80)
    print("Testing all metaheuristics with 1D problems")
    all_mh_results_1 = eval_with_all_mh(cls_1s, verbose=False)
    
    print("#"*80)
    print("Testing all metaheuristics with 2D problems")
    all_mh_results_2 = eval_with_all_mh(cls_2s, verbose=False)
    
    # Print summary results
    print("#"*80)
    print("SUMMARY OF ALL RESULTS")
    print("#"*80)
    
    for problem_type, results in [("n-dimensional", all_mh_results_n), 
                                  ("1-dimensional", all_mh_results_1), 
                                  ("2-dimensional", all_mh_results_2)]:
        print(f"\n{problem_type.upper()} PROBLEMS:")
        for mh_name, mh_results in results.items():
            if "error" in mh_results:
                print(f"  {mh_name}: ERROR - {mh_results['error']}")
            else:
                avg_diff = sum(result["diff"] for result in mh_results.values()) / len(mh_results)
                print(f"  {mh_name}: Average difference from optimum: {avg_diff:.6f}")
        print("-" * 40)
    
    # Create comprehensive plots comparing all metaheuristics
    print("\nGenerating comparison plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Average performance by problem type
    ax1 = plt.subplot(2, 3, 1)
    problem_types = ["n-dimensional", "1-dimensional", "2-dimensional"]
    all_results_list = [all_mh_results_n, all_mh_results_1, all_mh_results_2]
    
    avg_performances = {}
    for mh_name in metaheuristics:
        avg_performances[mh_name] = []
        for results in all_results_list:
            if mh_name in results and "error" not in results[mh_name]:
                avg_diff = sum(result["diff"] for result in results[mh_name].values()) / len(results[mh_name])
                avg_performances[mh_name].append(avg_diff)
            else:
                avg_performances[mh_name].append(float('inf'))  # Use inf for errors
    
    # Create bar plot for average performance
    x_pos = np.arange(len(problem_types))
    width = 0.07
    colors = plt.cm.tab20(np.linspace(0, 1, len(metaheuristics)))
    
    for i, (mh_name, performances) in enumerate(avg_performances.items()):
        # Filter out infinite values for plotting
        plot_performances = [p if p != float('inf') else 0 for p in performances]
        ax1.bar(x_pos + i * width, plot_performances, width, label=mh_name, color=colors[i])
    
    ax1.set_xlabel('Problem Types')
    ax1.set_ylabel('Average Difference from Optimum (log scale)')
    ax1.set_title('Average Performance by Problem Type')
    ax1.set_xticks(x_pos + width * (len(metaheuristics) - 1) / 2)
    ax1.set_xticklabels(problem_types)
    ax1.set_yscale('log')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heatmap of performance across all problems
    ax2 = plt.subplot(2, 3, 2)
    
    # Collect all problem names and create performance matrix
    all_problems = set()
    for results in all_results_list:
        for mh_results in results.values():
            if "error" not in mh_results:
                all_problems.update(mh_results.keys())
    
    all_problems = sorted(list(all_problems))
    performance_matrix = np.full((len(metaheuristics), len(all_problems)), np.nan)
    
    for i, mh_name in enumerate(metaheuristics):
        for results in all_results_list:
            if mh_name in results and "error" not in results[mh_name]:
                for j, problem in enumerate(all_problems):
                    if problem in results[mh_name]:
                        performance_matrix[i, j] = np.log10(results[mh_name][problem]["diff"] + 1e-10)
    
    # Create heatmap
    im = ax2.imshow(performance_matrix, cmap='viridis_r', aspect='auto')
    ax2.set_xticks(range(len(all_problems)))
    ax2.set_xticklabels([p.split('.')[-1] for p in all_problems], rotation=45, ha='right')
    ax2.set_yticks(range(len(metaheuristics)))
    ax2.set_yticklabels(metaheuristics)
    ax2.set_title('Performance Heatmap (log10 difference from optimum)')
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: Box plot of performances for n-dimensional problems
    ax3 = plt.subplot(2, 3, 3)
    box_data = []
    box_labels = []
    
    for mh_name in metaheuristics:
        if mh_name in all_mh_results_n and "error" not in all_mh_results_n[mh_name]:
            differences = [result["diff"] for result in all_mh_results_n[mh_name].values()]
            box_data.append(differences)
            box_labels.append(mh_name)
    
    if box_data:
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
    
    ax3.set_xlabel('Metaheuristics')
    ax3.set_ylabel('Difference from Optimum (log scale)')
    ax3.set_title('Performance Distribution (N-dimensional Problems)')
    ax3.set_yscale('log')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Success rate (problems solved within tolerance)
    ax4 = plt.subplot(2, 3, 4)
    tolerance = 1e-3  # Define success as difference < 1e-3
    
    success_rates = {}
    for mh_name in metaheuristics:
        total_problems = 0
        successful_problems = 0
        
        for results in all_results_list:
            if mh_name in results and "error" not in results[mh_name]:
                for problem_result in results[mh_name].values():
                    total_problems += 1
                    if problem_result["diff"] < tolerance:
                        successful_problems += 1
        
        if total_problems > 0:
            success_rates[mh_name] = (successful_problems / total_problems) * 100
        else:
            success_rates[mh_name] = 0
    
    mh_names = list(success_rates.keys())
    rates = list(success_rates.values())
    
    bars = ax4.bar(mh_names, rates, color=colors[:len(mh_names)])
    ax4.set_xlabel('Metaheuristics')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title(f'Success Rate (Difference < {tolerance})')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 5: Performance ranking across all problems
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate average rank for each metaheuristic
    rankings = {}
    for mh_name in metaheuristics:
        rankings[mh_name] = []
    
    # Rank metaheuristics for each problem
    for results in all_results_list:
        for problem in all_problems:
            problem_performances = {}
            for mh_name in metaheuristics:
                if (mh_name in results and "error" not in results[mh_name] and 
                    problem in results[mh_name]):
                    problem_performances[mh_name] = results[mh_name][problem]["diff"]
            
            if len(problem_performances) > 1:
                # Sort by performance (lower is better)
                sorted_mh = sorted(problem_performances.items(), key=lambda x: x[1])
                for rank, (mh_name, _) in enumerate(sorted_mh, 1):
                    rankings[mh_name].append(rank)
    
    # Calculate average rankings
    avg_rankings = {}
    for mh_name, ranks in rankings.items():
        if ranks:
            avg_rankings[mh_name] = np.mean(ranks)
        else:
            avg_rankings[mh_name] = len(metaheuristics)  # Worst possible rank
    
    # Sort by average ranking
    sorted_rankings = sorted(avg_rankings.items(), key=lambda x: x[1])
    mh_names_ranked = [x[0] for x in sorted_rankings]
    avg_ranks = [x[1] for x in sorted_rankings]
    
    bars = ax5.barh(mh_names_ranked, avg_ranks, color=colors[:len(mh_names_ranked)])
    ax5.set_xlabel('Average Rank (1 = best)')
    ax5.set_ylabel('Metaheuristics')
    ax5.set_title('Average Performance Ranking')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Convergence comparison (using theoretical optimum ratio)
    ax6 = plt.subplot(2, 3, 6)
    
    convergence_ratios = {}
    for mh_name in metaheuristics:
        ratios = []
        for results in all_results_list:
            if mh_name in results and "error" not in results[mh_name]:
                for problem_result in results[mh_name].values():
                    theory_opt = problem_result["theory"]
                    obtained_val = problem_result["val"]
                    if theory_opt != 0:
                        ratio = abs(obtained_val / theory_opt)
                    else:
                        ratio = abs(obtained_val) + 1  # Handle zero theoretical optimum
                    ratios.append(ratio)
        convergence_ratios[mh_name] = ratios
    
    # Create violin plot
    data_for_violin = []
    labels_for_violin = []
    for mh_name in metaheuristics:
        if mh_name in convergence_ratios and convergence_ratios[mh_name]:
            # Use log scale for better visualization
            log_ratios = np.log10(np.array(convergence_ratios[mh_name]) + 1e-10)
            data_for_violin.append(log_ratios)
            labels_for_violin.append(mh_name)
    
    if data_for_violin:
        parts = ax6.violinplot(data_for_violin, positions=range(len(labels_for_violin)))
        ax6.set_xticks(range(len(labels_for_violin)))
        ax6.set_xticklabels(labels_for_violin, rotation=45)
        ax6.set_ylabel('Log10(Obtained/Theoretical)')
        ax6.set_title('Convergence Quality Distribution')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metaheuristics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Plots saved as 'metaheuristics_comparison.png'")
    print("Analysis complete!")
