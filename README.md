# ANMetaL - Another Numeric optimization and Metaheuristics Library

**A comprehensive Python library for metaheuristic optimization, numeric methods, and NP-hard problem solving with powerful visualization capabilities.**

Whether you're optimizing complex functions, solving combinatorial problems, or researching metaheuristic algorithms, ANMetaL provides battle-tested implementations with beautiful visualizations.

## ✨ Key Features

### 🎬 Stunning Visualizations

ANMetaL now includes powerful visualization capabilities that create animated videos of metaheuristic optimization processes!

- **12 Population-based Algorithms**: ABC, ACO, AFSA, Bat, Black Hole, Cuckoo, Firefly, Harmony Search, PSO, Greedy, and more
- **2D & 3D Plotting**: Watch particles move in 2D space or explore 3D fitness landscapes
- **Automatic Video Generation**: Creates smooth MP4 animations with customizable frame rates
- **Multiple Test Functions**: Visualize optimization on various benchmark problems
- **Genetic Algorithm Visualization**: Track evolution on categorical problems like Knapsack and Sudoku

**Quick Start Visualization:**
```bash
anmetal mh_graph_each_it --mh Firefly --problem Goldsteinprice --iterations 20 --plottype 3d --fps 30
```

For complete documentation of all visualization commands and options, see the [CLI Documentation](#command-line-interface) section below.

### 🤖 12+ Metaheuristic Algorithms

**Population-Based Algorithms** (for continuous/real-valued problems):
- 🐝 **ABC** - Artificial Bee Colony
- 🐜 **ACO** - Ant Colony Optimization  
- 🐟 **AFSA** - Artificial Fish Swarm Algorithm
- 🦇 **Bat Algorithm** - Echolocation-based optimization
- ⚫ **Black Hole Algorithm** - Gravitational search
- 🐦 **Cuckoo Search** - Lévy flight optimization
- 🔥 **Firefly Algorithm** - Bio-luminescence inspired
- 🎵 **Harmony Search** - Music-inspired optimization
- 🌊 **PSO** - Particle Swarm Optimization (with Leap variant)
- 🎯 **Greedy Search** (with Leap variant)

**Evolutionary Algorithms** (for categorical/discrete problems):
- 🧬 **Genetic Algorithm** - Evolution-inspired optimization (with Leap variant)

### 📊 Comprehensive Problem Suite

**NP-Hard Problems:**
- 🎒 **Knapsack Problem** - Classic combinatorial optimization
- 🔢 **Sudoku Solver** - Constraint satisfaction (standard & optimized)
- ⚖️ **Partition Problem** - Subset partitioning
- ➕ **Subset Sum Problem** - Target sum finding

**Benchmark Functions** (20+ optimization test functions):
- **2D Functions**: Camelback, Goldstein-Price, Shubert variants, Quartic
- **N-Dimensional**: Sphere, Rosenbrock, Rastrigin, Griewank, Schwefel, and more
- **1D Functions**: F1, F3 test functions

### 🔧 Numeric Optimization Tools

**Single-Solution Iterative Methods:**
- Newton Method
- Euler Method

**Binary Optimization Utilities:**
- 9 binarization functions (sShape1-4, vShape1-4, erf)
- 4 binarization strategies (standard, complement, static_probability, elitist)

### 🎨 Flexible Visualization Options

- 2D and 3D plotting
- Multiple category visualization types (character, color, icon, number, value)
- Trajectory tracking and convergence analysis
- Automated video generation
- Customizable frame rates and plot styles

## 🚀 Installation

To install the core library:

```bash
pip install anmetal
```

For visualization and animation features, install with development dependencies:

```bash
pip install anmetal[test]
# or for development
pip install -e .[dev]
```

**Required dependencies for visualization:**
- `matplotlib` (plotting)
- `opencv-python` (video generation)
- `numpy` (numerical operations)

---

## 🛠️ Command Line Interface

ANMetaL provides a powerful CLI for running optimizations and creating visualizations. All commands follow the pattern:

```bash
anmetal <command> [arguments]
```

Available commands:
- `mh_graph_each_it` - Create animated visualizations of metaheuristic optimization
- `genetic_categorical_plot` - Visualize genetic algorithm on categorical problems
- `trajectory_plot_npcomp` - Plot trajectories for NP-complete problems
- `nonlinear_mh_optimization` - Run metaheuristics on nonlinear functions
- `basic_mh_optimization_npcomp` - Run metaheuristics on NP-hard problems
- `basic_math_it_optimization` - Test iterative optimization methods
- `categorical_mh_optimization` - Run genetic algorithms on categorical problems

See `/pytests` folder for additional examples of programmatic use.

## 📖 Quick Start Examples

### Basic Optimization

```python
# Example with Partition Problem
from anmetal.problems.nphard_real import Partition_Real
from anmetal.optimizer.population.PSO.PSOMH_Real import PSOMH_Real

# Create problem instance
problem = Partition_Real(seed=0, num_dims=200)

# Create and run metaheuristic
mh = PSOMH_Real(problem.min_x, problem.max_x, problem.ndim, False,
                problem.objective_function, problem.repair_function,
                problem.preprocess_function)

# Run optimization
fitness, solution = mh.run(verbose=True, iterations=100, population=30,
                         omega=0.8, phi_g=1, phi_p=0.5, seed=115)
```

### Visualization and Animation

```python
# Run from command line for visualization
anmetal mh_graph_each_it --mh PSO --problem Goldsteinprice --iterations 50 --plottype 3d

# This creates:
# - Individual iteration images
# - MP4 animation video
# - Real-time optimization visualization
```

For more detailed command-line options, see the [Command Line Interface](#command-line-interface) section.

## 📚 Complete API Reference

### Optimization Methods

#### Single-Solution Iterative Methods
- **Newton Method** - Root finding and optimization
- **Euler Method** - Numerical differential equation solving

### Metaheuristic Algorithms

#### Population-Based (Real/Continuous Input)
| Algorithm | Description | Variants |
|-----------|-------------|----------|
| **ABC** | Artificial Bee Colony | Standard |
| **ACO** | Ant Colony Optimization | Standard |
| **AFSA** | Artificial Fish Swarm Algorithm | Standard |
| **Bat** | Bat Algorithm | Standard |
| **Black Hole** | Black Hole Algorithm | Standard |
| **Cuckoo** | Cuckoo Search | Standard |
| **Firefly** | Firefly Algorithm | Standard |
| **Harmony Search** | Harmony Search | Standard |
| **PSO** | Particle Swarm Optimization | Standard, With Leap |
| **Greedy** | Greedy Search | Standard, With Leap |

#### Evolutionary (Categorical/Discrete Input)
| Algorithm | Description | Variants |
|-----------|-------------|----------|
| **Genetic Algorithm** | Evolution-based optimization | Standard, With Leap |

### Test Problems

#### NP-Hard Problems

**Real-Valued:**
- **Partition Problem** - Divide set into equal-sum subsets
- **Subset Sum Problem** - Find subset with target sum

**Categorical:**
- **Knapsack Problem** - Maximize value within weight constraint
- **Sudoku** - Fill 9×9 grid with constraints (standard & optimized versions)

#### Benchmark Functions

**1-Dimensional Functions:**
- F1, F3 ([Ref](https://doi.org/10.1007/s00521-017-3088-3))

**2-Dimensional Functions:**
- Camelback, Goldstein-Price, Pshubert1, Pshubert2, Shubert, Quartic ([Ref](https://doi.org/10.1007/s00521-017-3088-3))

**N-Dimensional Functions:**
- Brown1, Brown3, F10n, F15n ([Ref](https://doi.org/10.1007/s00521-017-3088-3))
- Sphere, Rosenbrock, Griewank, Rastrigin, Sumsquares, Michalewicz, Quartic, Schwefel, Penalty ([Ref](https://doi.org/10.1007/s00521-018-3512-3))

### Utility Functions

#### Binarization Functions
Transform continuous values to binary for discrete optimization:
- **S-Shape family**: sShape1, sShape2, sShape3, sShape4
- **V-Shape family**: vShape1, vShape2, vShape3, vShape4
- **Error function**: erf

#### Binarization Strategies
Methods for applying binarization in optimization:
- **standard** - Basic threshold-based conversion
- **complement** - Inverted threshold
- **static_probability** - Fixed probability approach
- **elitist** - Elite-preserving conversion

---

## 🛠️ Command Line Interface

#### 1. Metaheuristic Animation (`mh_graph_each_it`)

Creates animated visualizations of population-based metaheuristic optimization processes with 2D or 3D plots.

**Basic Usage:**
```bash
anmetal mh_graph_each_it --mh <algorithm> --problem <function> [options]
```

**Common Arguments:**
- `--folder <path>` - Output folder for plots and videos (default: `mh_graphs`)
- `--seed <int>` - Random seed for reproducibility (default: 0)
- `--mh <algorithm>` - Algorithm to use (required)
  - Options: `AFSA`, `PSO`, `PSOWL`, `Greed`, `GreedWL`, `ABC`, `ACO`, `Bat`, `Blackhole`, `Cuckoo`, `Firefly`, `Harmony`
- `--problem <function>` - Test function (required)
  - Options: `Camelback`, `Goldsteinprice`, `Pshubert1`, `Pshubert2`, `Shubert`, `Quartic`
- `--verbose <0|1>` - Print optimization logs (default: 1)
- `--iterations <int>` - Number of iterations (default: 100)
- `--population <int>` - Population size (default: 30)
- `--plottype <2d|3d>` - Plot type (default: `2d`)
- `--traces <all|none|smooth>` - Trace visualization mode (default: `none`)
- `--fps <int>` - Video frame rate (default: 10)

**Algorithm-Specific Parameters:**

*AFSA (Artificial Fish Swarm Algorithm):*
```bash
--parameter-afsa-visualdistancepercentage <float>  # Visual distance (default: 0.2)
--parameter-afsa-velocitypercentage <float>        # Velocity (default: 0.3)
--parameter-afsa-npointstochoose <int>            # Points to choose (default: 5)
--parameter-afsa-crowdedpercentage <float>        # Crowded threshold (default: 0.8)
--parameter-afsa-itsstagnation <int>              # Stagnation iterations (default: 7)
--parameter-afsa-leappercentage <float>           # Leap distance (default: 0.2)
--parameter-afsa-stagnationvariation <float>      # Variation threshold (default: 0.4)
```

*PSO (Particle Swarm Optimization):*
```bash
--parameter-pso-omega <float>   # Inertia weight (default: 0.5)
--parameter-pso-phig <float>    # Global coefficient (default: 1.0)
--parameter-pso-phip <float>    # Personal coefficient (default: 2.0)
```

*PSOWL (PSO with Leap):*
```bash
--parameter-psowl-omega <float>                # Inertia weight (default: 0.5)
--parameter-psowl-phig <float>                 # Global coefficient (default: 1.0)
--parameter-psowl-phip <float>                 # Personal coefficient (default: 2.0)
--parameter-psowl-stagnationvariation <float>  # Variation threshold (default: 0.4)
--parameter-psowl-itsstagnation <int>          # Stagnation iterations (default: 5)
--parameter-psowl-leappercentage <float>       # Leap distance (default: 0.8)
```

*GreedWL (Greedy with Leap):*
```bash
--parameter-greedwl-stagnationvariation <float>  # Variation threshold (default: 0.4)
--parameter-greedwl-itsstagnation <int>          # Stagnation iterations (default: 5)
--parameter-greedwl-leappercentage <float>       # Leap distance (default: 0.8)
```

*ABC (Artificial Bee Colony):*
```bash
--parameter-abc-limit <int>  # Abandonment limit (default: 20)
```

*ACO (Ant Colony Optimization):*
```bash
--parameter-aco-evaporationrate <float>  # Pheromone evaporation (default: 0.1)
--parameter-aco-alpha <float>            # Pheromone importance (default: 1.0)
--parameter-aco-beta <float>             # Heuristic importance (default: 2.0)
```

*BAT Algorithm:*
```bash
--parameter-bat-fmin <float>  # Minimum frequency (default: 0.0)
--parameter-bat-fmax <float>  # Maximum frequency (default: 2.0)
--parameter-bat-a <float>     # Loudness (default: 0.9)
--parameter-bat-r0 <float>    # Pulse rate (default: 0.9)
```

*CUCKOO Search:*
```bash
--parameter-cuckoo-pa <float>  # Abandonment probability (default: 0.25)
```

*FIREFLY Algorithm:*
```bash
--parameter-firefly-alpha <float>  # Randomness (default: 0.5)
--parameter-firefly-beta0 <float>  # Attractiveness (default: 1.0)
--parameter-firefly-gamma <float>  # Absorption coefficient (default: 1.0)
```

*HARMONY Search:*
```bash
--parameter-harmony-hmcr <float>  # Harmony Memory Considering Rate (default: 0.9)
--parameter-harmony-par <float>   # Pitch Adjustment Rate (default: 0.3)
--parameter-harmony-bw <float>    # Bandwidth (default: 0.2)
```

**Examples:**
```bash
# 2D visualization with ABC algorithm
anmetal mh_graph_each_it --mh ABC --problem Camelback --iterations 20 --population 30

# 3D visualization with custom parameters
anmetal mh_graph_each_it --mh Firefly --problem Goldsteinprice --iterations 15 --plottype 3d --fps 30 --parameter-firefly-alpha 0.6

# PSO with traces and custom parameters
anmetal mh_graph_each_it --mh PSO --problem Quartic --traces all --parameter-pso-omega 0.7 --parameter-pso-phig 1.5

# Custom output folder
anmetal mh_graph_each_it --mh ABC --problem Shubert --folder ./my_results --iterations 50
```

#### 2. Genetic Algorithm Visualization (`genetic_categorical_plot`)

Visualize genetic algorithms on categorical problems like Knapsack and Sudoku.

**Basic Usage:**
```bash
anmetal genetic_categorical_plot --problem <problem> [options]
```

**Arguments:**
- `--folder <path>` - Output folder for plots and videos (default: `mh_graphs`)
- `--seed <int>` - Random seed (default: 0)
- `--iterations <int>` - Number of generations (default: 50)
- `--population <int>` - Population size (default: 20)
- `--elitist <float>` - Elitist percentage (default: 0.3)
- `--mutability <float>` - Mutation rate (default: 0.1)
- `--fidelity <0|1>` - Enable fidelity (default: 1)
- `--parents_mutation <0|1>` - Enable parent mutation (default: 1)
- `--leap <0|1>` - Use leap variant (default: 0)
- `--categorytype <type>` - Visualization type (default: `character`)
  - Options: `character`, `color`, `icon`, `number`, `value`, `colorvalue`
- `--fps <int>` - Video frame rate (default: 5)
- `--problem <problem>` - Problem to solve (required)
  - Options: `knapsack`, `sudoku`, `sudoku_opt`
- `--dims <int>` - Problem dimensions (for Knapsack, default: 20)

**Examples:**
```bash
# Knapsack with color visualization
anmetal genetic_categorical_plot --problem knapsack --dims 30 --categorytype color --iterations 100

# Sudoku with optimized version
anmetal genetic_categorical_plot --problem sudoku_opt --categorytype colorvalue --iterations 200 --population 50

# With leap variant
anmetal genetic_categorical_plot --problem knapsack --leap 1 --mutability 0.2 --elitist 0.4
```

#### 3. NP-Complete Trajectory Plotting (`trajectory_plot_npcomp`)

Plot fitness trajectories of metaheuristics on NP-complete problems (Partition and Subset Sum).

**Basic Usage:**
```bash
anmetal trajectory_plot_npcomp --algorithms <alg1> <alg2> ... [options]
```

**Arguments:**
- `--folder <path>` - Output folder for plots (default: current directory)
- `--algorithms <list>` - Algorithms to run (space-separated, required)
  - Options: `AFSA`, `Greedy`, `GreedyWL`, `PSO`, `PSOWL`, `ABC`, `ACO`, `BAT`, `BH`, `CUCKOO`, `FIREFLY`, `HS`
  - Default: `PSO ABC ACO BAT FIREFLY HS`
- `--problem <problem>` - Problem type (default: `partition sum`)
  - Options: `partition sum`, `subset sum`
- `--verbose <0|1|2>` - Verbosity level (default: 0)
  - 0=quiet, 1=normal, 2=debug
- `--max-iterations <int>` - Maximum iterations (default: 50)
- `--population-size <int>` - Population size (default: 20)
- `--seed <int>` - Random seed (default: 115)
- `--ndims <int>` - Problem dimensions (default: 500)

**Algorithm-Specific Parameters:**
All algorithm parameters from `mh_graph_each_it` are supported with the same names.

**Examples:**
```bash
# Compare PSO and ABC on partition problem
anmetal trajectory_plot_npcomp --algorithms PSO ABC --problem "partition sum" --max-iterations 100

# Run AFSA with custom parameters
anmetal trajectory_plot_npcomp --algorithms AFSA --parameter-afsa-visualdistancepercentage 0.6 --ndims 1000

# Multiple algorithms with custom settings
anmetal trajectory_plot_npcomp --algorithms PSO ABC ACO --folder ./results --ndims 1000 --seed 42 --verbose 1
```

#### 4. Nonlinear Function Optimization (`nonlinear_mh_optimization`)

Run metaheuristics on various nonlinear benchmark functions. This is a Python script for testing and comparison - no CLI arguments.

#### 5. Basic Optimization Scripts

The following commands are available for testing but don't require specific arguments (they use hardcoded test cases):

- **`basic_mh_optimization_npcomp`** - Test metaheuristics on Partition/Subset Sum problems
- **`basic_math_it_optimization`** - Test Newton and Euler iterative methods
- **`categorical_mh_optimization`** - Test genetic algorithms on categorical problems

**Usage:**
```bash
anmetal basic_mh_optimization_npcomp
anmetal basic_math_it_optimization
anmetal categorical_mh_optimization
```

---

## 🎛️ Algorithm Parameters

Each metaheuristic has its own set of parameters. Here are some common ones:

* **Common Parameters**
  * `iterations`: Number of iterations
  * `population`: Population size
  * `seed`: Random seed for reproducibility
  * `verbose`: Whether to print progress

* **Algorithm-Specific Parameters**
  * ABC: `limit`
  * ACO: `evaporation_rate`, `alpha`, `beta`
  * BAT: `fmin`, `fmax`, `A`, `r0`
  * CUCKOO: `pa`
  * FIREFLY: `alpha`, `beta0`, `gamma`
  * GA: `mutation_rate`, `crossover_rate`
  * HS: `hmcr`, `par`, `bw`
  * PSO: `omega`, `phi_g`, `phi_p`

For detailed parameter descriptions and recommended values, see the respective algorithm implementations in the source code.

## Troubleshooting

### Visualization Issues

**OpenCV Installation Problems:**
```bash
# If you encounter OpenCV issues, try:
pip install opencv-python-headless
```

**Missing Dependencies:**
```bash
# Install all visualization dependencies:
pip install matplotlib opencv-python numpy seaborn pandas
```

**Video Generation Errors:**
- Ensure you have write permissions in the output directory
- Check that OpenCV is properly installed
- Try reducing FPS if encountering codec issues

### Algorithm-Specific Issues

**Parameter Tuning:**
- Start with default parameters and adjust gradually
- Use smaller populations for testing (5-15 individuals)
- Reduce iterations for quick tests (5-20 iterations)

**Performance:**
- Use `--verbose 0` to reduce console output
- Lower FPS for faster video generation
- Consider 2D plotting for better performance
