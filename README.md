# ANMetaL - Another Numeric optimization and Metaheuristics Library

**A zero-dependency TypeScript library for metaheuristic optimization, numeric methods, and NP-hard problem solving. Compiles to a single `.min.js` file for browser use.**

## ✨ Key Features

- **Zero runtime dependencies** — all math is native JS
- **Browser-ready** — ships as a single UMD `anmetal.min.js` (53KB)
- **ESM support** — also available as `anmetal.esm.js`
- **12+ Metaheuristic Algorithms** with generator-based iteration tracking
- **23+ Benchmark Functions** and **5 NP-Hard Problems**
- **Seeded PRNG** for reproducible results

## 🚀 Installation

```bash
npm install anmetal
```

### Browser (via `<script>`)

```html
<script src="dist/anmetal.min.js"></script>
<script>
  const result = ANMetaL.PSOMH_Real(/* ... */);
</script>
```

### ES Module

```javascript
import { PSOMH_Real, Goldsteinprice } from "anmetal";
```

## 📖 Quick Start

### PSO on Goldstein-Price

```typescript
import { PSOMH_Real, Goldsteinprice } from "anmetal";

const limits = Goldsteinprice.getLimits();
const pso = new PSOMH_Real(
  (limits as any).min[0], (limits as any).max[0],
  2, false,
  (p: number[]) => Goldsteinprice.func(p),
  (p: number[]) => p,
  (p: number[]) => p
);

const { fitness, point } = pso.run(100, 30, 42, 0.5, 1, 1);
console.log(`Best fitness: ${fitness}, point: ${point}`);
```

### Track Iterations with `runYielded()`

```typescript
const gen = pso.runYielded(100, 30, 42, 0.5, 1, 1);
for (const step of gen) {
  console.log(`Iteration ${step.iteration}: fitness=${step.bestFitness}`);
}
```

### Genetic Algorithm on Knapsack

```typescript
import { GeneticMH_Categorical, KnapsackCategorical } from "anmetal";

const problem = new KnapsackCategorical(50, 20, 42, 10, 10);
const ga = new GeneticMH_Categorical(
  problem.getPossibleCategories(),
  20, true,
  (p) => problem.objectiveFunction(p),
  (p) => problem.repairFunction(p),
  (p) => problem.preprocessFunction(p)
);
const { fitness, point } = ga.run(50, 20, 0.3, 0.1, true, true, 42);
```

### Newton's Method

```typescript
import { newtonMethod } from "anmetal";

const root = newtonMethod(
  (x) => x * x - 4,
  (x) => 2 * x,
  3, 20
);
// root ≈ 2.0
```

## 🤖 Algorithms

### Population-Based (Real/Continuous)

| Class | Algorithm | Variants |
|-------|-----------|----------|
| `PSOMH_Real` | Particle Swarm Optimization | Standard |
| `PSOMH_Real_WithLeap` | PSO with Leap | WithLeap extends PSO |
| `GreedyMH_Real` | Greedy Search | Standard |
| `GreedyMH_Real_WithLeap` | Greedy with Leap | WithLeap extends Greedy |
| `AFSAMH_Real` | Artificial Fish Swarm | — |
| `ArtificialBeeColony` | Artificial Bee Colony | — |
| `AntColony` | Ant Colony Optimization | — |
| `BatAlgorithm` | Bat Algorithm | — |
| `BlackHole` | Black Hole Algorithm | — |
| `CuckooSearch` | Cuckoo Search | — |
| `FireflyAlgorithm` | Firefly Algorithm | — |
| `HarmonySearch` | Harmony Search | — |

### Evolutionary (Categorical/Discrete)

| Class | Algorithm | Variants |
|-------|-----------|----------|
| `GeneticMH_Categorical` | Genetic Algorithm | Standard |
| `GeneticMH_Categorical_WithLeap` | GA with Leap | WithLeap extends GA |

### Single-Solution Methods

| Function | Description |
|----------|-------------|
| `eulerMethod(dydx, x0, xEnd, y0, steps)` | Numerical ODE integration |
| `newtonMethod(f, df, x0, iters, m?)` | Newton-Raphson root finding |

## 📊 Problems

### Benchmark Functions

**1-Dimensional:** `F1`, `F3`

**2-Dimensional:** `Camelback`, `Goldsteinprice`, `Pshubert1`, `Pshubert2`, `Shubert`, `Quartic2D`

**N-Dimensional:** `Sphere`, `Rosenbrock`, `Griewank`, `Rastrigrin`, `Sumsquares`, `Michalewicz`, `Quartic`, `Schwefel`, `Penalty`, `Brown1`, `Brown3`, `F10n`, `F15n`

Each function class provides:
- `static func(x: number[]): number`
- `static getLimits(): { min, max }`
- `static getTheoreticalOptimum(): number`
- `static getType(): string`

### NP-Hard Problems

| Class | Problem | Type |
|-------|---------|------|
| `PartitionReal` | Partition Problem | Real-valued |
| `SubsetReal` | Subset Sum Problem | Real-valued |
| `KnapsackCategorical` | 0/1 Knapsack | Categorical |
| `Sudoku` | Sudoku Solver | Validation |
| `SudokuOptimized` | Sudoku (optimized) | Categorical |

## 🔧 Utilities

### SeededRandom

```typescript
import { SeededRandom } from "anmetal";

const rng = new SeededRandom(42);
rng.next();                    // [0, 1)
rng.nextFloat(-5, 5);         // uniform float
rng.nextInt(0, 10);           // integer
rng.normal(0, 1);             // Gaussian
rng.shuffle([1, 2, 3, 4]);   // Fisher-Yates
rng.choice([1, 2, 3]);       // random element
rng.choiceWeighted([0.5, 0.3, 0.2]); // weighted index
rng.randomArray(10, 0, 1);   // array of floats
```

### Binarization Functions

```typescript
import { sShape1, vShape1, erf } from "anmetal";
sShape1(0); // 0.5
vShape1(0); // 0
```

Full set: `sShape1-4`, `vShape1-4`, `erf`

### Binarization Strategies

```typescript
import { standard, complement, staticProbability, elitist } from "anmetal";
standard(0.7, 0.5); // 1
```

### Point Utilities

```typescript
import { distance, distanceSquared, distanceTaxicab, nsphereToCartesian } from "anmetal";
distance([0, 0], [3, 4]); // 5
```

## 🎛️ Algorithm Parameters

### Common

| Parameter | Description |
|-----------|-------------|
| `iterations` | Number of iterations |
| `population` | Population size |
| `seed` | Random seed for reproducibility |
| `verbose` | Print progress (where supported) |

### Algorithm-Specific

| Algorithm | Parameters |
|-----------|------------|
| **PSO** | `omega`, `phi_p`, `phi_g` |
| **PSOWL** | + `stagnation_variation`, `its_stagnation`, `leap_percentage` |
| **GreedyWL** | `stagnation_variation`, `its_stagnation`, `leap_percentage` |
| **AFSA** | `visual_distance_percentage`, `velocity_percentage`, `n_points_to_choose`, `crowded_percentage`, `stagnation_variation`, `its_stagnation`, `leap_percentage` |
| **ABC** | `limit` |
| **ACO** | `evaporation_rate`, `alpha`, `beta` |
| **Bat** | `fmin`, `fmax`, `A`, `r0` |
| **Cuckoo** | `pa` |
| **Firefly** | `alpha`, `beta0`, `gamma` |
| **Harmony** | `hmcr`, `par`, `bw` |
| **Genetic** | `elitist_percentage`, `mutability`, `fidelity`, `mutation_in_parents` |
| **GeneticWL** | + `stagnation_variation`, `its_stagnation`, `leap_percentage` |

## 🛠️ Build & Test

```bash
npm run build    # Compile to dist/anmetal.min.js + dist/anmetal.esm.js
npm test         # Build + run 81 vitest tests against compiled output
```

## 📁 Project Structure

```
src/
  index.ts                  # Barrel export (public API)
  optimizer/
    IMetaheuristic.ts       # Abstract base + YieldResult/RunResult types
    ISolution.ts            # SolutionBasic, SolutionWithId, SolutionParticle
    single_solution/
      iterativeMethods.ts   # eulerMethod, newtonMethod
    population/
      pso/                  # PSO + PSOWL
      greedy/               # Greedy + GreedyWL
      afsa/                 # AFSA
      abc/                  # Artificial Bee Colony
      aco/                  # Ant Colony
      bat/                  # Bat Algorithm
      blackhole/            # Black Hole
      cuckoo/               # Cuckoo Search
      firefly/              # Firefly Algorithm
      harmony/              # Harmony Search
      genetic/              # Genetic + GeneticWL
  problems/
    IProblem.ts             # Abstract problem interface
    nonlinear_functions/    # 23 benchmark functions
    nphard_real/            # Partition, Subset Sum
    nphard_categorical/     # Knapsack, Sudoku, SudokuOptimized
  utils/
    random.ts               # SeededRandom (mulberry32 PRNG)
    pointsUtils.ts          # Distance, nsphere utilities
    binarizationFloat.ts    # 9 transfer functions
    binarizationStrategy.ts # 4 binarization strategies
tests/                      # Vitest tests (run against compiled .min.js)
dist/                       # Build output
  anmetal.min.js            # 53KB UMD bundle
  anmetal.esm.js            # ESM bundle
```

## Legacy Python Version

The original Python implementation is preserved in `old/`. CLI commands (`mh_graph_each_it`, `genetic_categorical_plot`, etc.) will be converted to HTML files in a future update.

## License

MIT
