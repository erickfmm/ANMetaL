# Another Numeric optimization and Metaheuristics Library

A library to do your metaheuristics and numeric combinatorial stuff.

## 🎬 Visualization Highlights

ANMetaL now includes powerful visualization capabilities that create animated videos of metaheuristic optimization processes!

- **12 Population-based Algorithms**: ABC, ACO, AFSA, Bat, Black Hole, Cuckoo, Firefly, Harmony Search, PSO, Greedy, and more
- **2D & 3D Plotting**: Watch particles move in 2D space or explore 3D fitness landscapes
- **Automatic Video Generation**: Creates smooth MP4 animations with customizable frame rates
- **Multiple Test Functions**: Visualize optimization on various benchmark problems

**Quick Start Visualization:**
```bash
python test/mh_graph_each_it.py --mh Firefly --problem Goldsteinprice --iterations 20 --plot3d 1 --fps 30
```

## Installation

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

See `/test` folder for examples of use.

## Content

### Numeric optimization
Iterative optimization functions (one solution)
* Euler method
* Newton method

### Metaheuristics

#### Real input (Population-based)
* Artificial Bee Colony (ABC)
* Ant Colony Optimization (ACO)
* Artificial Fish Swarm Algorithm (AFSA)
* Bat Algorithm
* Blackhole Algorithm
* Cuckoo Search
* Firefly Algorithm
* Harmony Search (HS)
* Particle Swarm Optimization (PSO)
* Particle Swarm Optimization with Leap
* Greedy
* Greedy with Leap

#### Categorical input
* Genetic Algorithm
* Genetic Algorithm with Leap

### Problems and gold-standard functions

#### NP-hard problems

* Real problems
  * Partition problem
  * Subset problem

* Categorical problems
  * Knapsack
  * Sudoku (without initial matrix, just random)

#### Non linear functions

* One input (1-D)
  * F1 (https://doi.org/10.1007/s00521-017-3088-3)
  * F3 (https://doi.org/10.1007/s00521-017-3088-3)

* Two inputs (2-D)
  * Camelback (https://doi.org/10.1007/s00521-017-3088-3)
  * Goldsteinprice (https://doi.org/10.1007/s00521-017-3088-3)
  * Pshubert1 (https://doi.org/10.1007/s00521-017-3088-3)
  * Pshubert2 (https://doi.org/10.1007/s00521-017-3088-3)
  * Shubert (https://doi.org/10.1007/s00521-017-3088-3)
  * Quartic (https://doi.org/10.1007/s00521-017-3088-3)

* N inputs (N-D)
  * Brown1 (https://doi.org/10.1007/s00521-017-3088-3)
  * Brown3 (https://doi.org/10.1007/s00521-017-3088-3)
  * F10n (https://doi.org/10.1007/s00521-017-3088-3)
  * F15n (https://doi.org/10.1007/s00521-017-3088-3)
  * Sphere (https://doi.org/10.1007/s00521-018-3512-3)
  * Rosenbrock (https://doi.org/10.1007/s00521-018-3512-3)
  * Griewank (https://doi.org/10.1007/s00521-018-3512-3)
  * Rastrigrin (https://doi.org/10.1007/s00521-018-3512-3)
  * Sumsquares (https://doi.org/10.1007/s00521-018-3512-3)
  * Michalewicz (https://doi.org/10.1007/s00521-018-3512-3)
  * Quartic (https://doi.org/10.1007/s00521-018-3512-3)
  * Schwefel (https://doi.org/10.1007/s00521-018-3512-3)
  * Penalty (https://doi.org/10.1007/s00521-018-3512-3)

### Additional Features

#### Metaheuristic Visualization and Animation

ANMetaL includes a powerful visualization tool (`test/mh_graph_each_it.py`) that creates animated visualizations of metaheuristic optimization processes:

**Features:**
- **2D and 3D Plotting**: Visualize population movement in 2D space or 3D with fitness as Z-axis
- **Video Generation**: Automatically creates MP4 animations from image sequences
- **All Population Algorithms**: Supports all population-based metaheuristics (except genetic algorithm)
- **Multiple Problems**: Works with various 2D optimization functions

**Supported Algorithms:**
- AFSA (Artificial Fish Swarm Algorithm)
- PSO (Particle Swarm Optimization) and PSO with Leap
- ABC (Artificial Bee Colony)
- ACO (Ant Colony Optimization)
- Bat Algorithm
- Black Hole Algorithm
- Cuckoo Search
- Firefly Algorithm
- Harmony Search
- Greedy and Greedy with Leap

**Usage Examples:**
```bash
# 2D visualization with ABC algorithm
python test/mh_graph_each_it.py --mh ABC --problem Camelback --iterations 20 --population 30

# 3D visualization with fitness landscape
python test/mh_graph_each_it.py --mh Firefly --problem Goldsteinprice --iterations 15 --plot3d 1 --fps 30

# Quick test with custom parameters
python test/mh_graph_each_it.py --mh Bat --problem Shubert --iterations 10 --plot3d 1 --fps 20 --seed 42
```

**Available Arguments:**
- `--mh`: Algorithm (AFSA, PSO, PSOWL, ABC, ACO, Bat, Blackhole, Cuckoo, Firefly, Harmony, Greed, GreedWL)
- `--problem`: Test function (Camelback, Goldsteinprice, Pshubert1, Pshubert2, Shubert, Quartic)
- `--iterations`: Number of optimization iterations
- `--population`: Population size
- `--plot3d`: Enable 3D plotting (0 for 2D, 1 for 3D)
- `--fps`: Video frame rate
- `--seed`: Random seed for reproducibility
- `--verbose`: Print optimization progress

**Output:**
The visualization tool creates:
- Individual PNG images for each iteration in `mh_graphs/{algorithm}_{problem}/`
- An MP4 video animation: `{algorithm}_{problem}_animation.mp4`
- Console output showing optimization progress and best fitness values

#### Binarization functions
* sShape1
* sShape2
* sShape3
* sShape4
* vShape1
* vShape2
* vShape3
* vShape4
* erf

#### Binarization strategies
* standard
* complement
* static_probability
* elitist

## Example Usage

See the `/test` folder for complete examples. Here's a quick overview:

### Basic Optimization

```python
# Example with Partition Problem
from anmetal.problems.nphard_real import Partition_Real
from anmetal.population.PSO.PSOMH_Real import PSOMH_Real

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
python test/mh_graph_each_it.py --mh PSO --problem Goldsteinprice --iterations 50 --plot3d 1

# This creates:
# - Individual iteration images
# - MP4 animation video
# - Real-time optimization visualization
```

## Algorithm Parameters

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
