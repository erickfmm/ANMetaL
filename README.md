# Another Numeric optimization and Metaheuristics Library

A library to do your metaheuristics and numeric combinatorial stuff.

to install use

´´´
pip install anmetal
´´´

see /test folder to see some examples of use

In later updates I will make documentation, for now there is only examples code


## Content

### Numeric optimization
Iterative optimization functions (one solution)
* Euler method
* Newton method


### Metaheuristics

Real input
* Artificial Fish Swarm Algorithm (AFSA) (Li, X. L. (2003). A new intelligent optimization-artificial fish swarm algorithm. Doctor thesis, Zhejiang University of Zhejiang, China, 27.)
* Particle Swarm Optimization (PSO) (Based on https://en.wikipedia.org/wiki/Particle_swarm_optimization)
* Particle Swarm Optimization (PSO) With Leap
* Greedy
* Greedy With Leap

Categorical input
* Genetic
* Genetic With Leap

### Problems and gold-standard functions

Nphard problems

* Real problems
  * Partition problem
  * Subset problem

* Categorical problems
  * knapsack
  * sudoku (without initial matrix, just random)

Non linear functions

* one input (1-D)
  * F1 (https://doi.org/10.1007/s00521-017-3088-3)
  * F3 (https://doi.org/10.1007/s00521-017-3088-3)
* two inputs (2-D)
  * Camelback (https://doi.org/10.1007/s00521-017-3088-3)
  * Goldsteinprice (https://doi.org/10.1007/s00521-017-3088-3)
  * Pshubert1 (https://doi.org/10.1007/s00521-017-3088-3)
  * Pshubert2 (https://doi.org/10.1007/s00521-017-3088-3)
  * Shubert (https://doi.org/10.1007/s00521-017-3088-3)
  * Quartic (https://doi.org/10.1007/s00521-017-3088-3)
* n inputs (N-D)
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

### Another content

Binarization functions
* sShape1
* sShape2
* sShape3
* sShape4
* vShape1
* vShape2
* vShape3
* vShape4
* erf

Binarization strategies
* standard
* complement
* static_probability
* elitist
