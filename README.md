# Another Numeric optimization and Metaheuristics Library

A library to do your metaheuristics and numeric combinatorial stuff.

To install use:

```bash
pip install anmetal
```

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

See the `/test` folder for complete examples. Here's a quick overview of running different metaheuristics:

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
