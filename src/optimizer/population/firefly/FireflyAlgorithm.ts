import { IMetaheuristic, YieldResult, RunResult } from "../../IMetaheuristic";
import { SolutionBasic } from "../../ISolution";
import { SeededRandom } from "../../../utils/random";

export class FireflyAlgorithm extends IMetaheuristic {
  private _min: number;
  private _max: number;
  private _ndims: number;
  private _rng!: SeededRandom;
  private _alpha: number = 0.5;
  private _beta0: number = 1.0;
  private _gamma: number = 1.0;
  private _fireflies: SolutionBasic[] = [];
  private _bestSolution: SolutionBasic | null = null;

  constructor(
    minValue: number,
    maxValue: number,
    ndims: number,
    toMax: boolean,
    objectiveFunction: (point: number[]) => number | false,
    repairFunction: (point: number[]) => number[],
    preprocessFunction?: (point: number[]) => number[]
  ) {
    super(toMax);
    this._min = minValue;
    this._max = maxValue;
    this._ndims = ndims;
    this.objectiveFunction = objectiveFunction;
    this.repairFunction = repairFunction;
    this.preprocessFunction = preprocessFunction ?? ((p) => p);
  }

  initializePopulation(populationSize: number): void {
    this._fireflies = [];
    for (let i = 0; i < populationSize; i++) {
      const point = this._rng.randomArray(this._ndims, this._min, this._max);
      const repaired = this.repairFunction(point);
      const preprocessed = this.preprocessFunction(repaired);
      const fitness = this.objectiveFunction(preprocessed);
      if (fitness !== false && fitness !== undefined && fitness !== null) {
        this._fireflies.push(new SolutionBasic(point, fitness));
      }
    }
    if (this._fireflies.length > 0) {
      this._bestSolution = this.findBestSolution(this._fireflies);
    }
  }

  private _distance(f1: SolutionBasic, f2: SolutionBasic): number {
    let sum = 0;
    for (let i = 0; i < f1.point.length; i++) {
      const diff = f1.point[i] - f2.point[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  private _attractAndMove(firefly1: SolutionBasic, firefly2: SolutionBasic): number[] {
    const distance = this._distance(firefly1, firefly2);
    const beta = this._beta0 * Math.exp(-this._gamma * distance * distance);
    const newPosition: number[] = new Array(this._ndims);
    for (let i = 0; i < this._ndims; i++) {
      const rand = this._rng.nextFloat(-0.5, 0.5);
      const movement = beta * (firefly2.point[i] - firefly1.point[i]) + this._alpha * rand;
      const pos = firefly1.point[i] + movement;
      newPosition[i] = Math.max(this._min, Math.min(this._max, pos));
    }
    return newPosition;
  }

  *runYielded(
    iterations: number = 100,
    population: number = 30,
    alpha: number = 0.5,
    beta0: number = 1.0,
    gamma: number = 1.0,
    seed?: number,
    verbose: boolean = false
  ): Generator<YieldResult> {
    this._iterations = iterations;
    this._alpha = alpha;
    this._beta0 = beta0;
    this._gamma = gamma;
    this._rng = new SeededRandom(seed ?? 0);

    this.initializePopulation(population);

    if (!this._fireflies || !this._bestSolution) {
      return;
    }

    let iteration = 1;
    let bestFitnessHistorical = this._bestSolution.fitness;
    let bestPointHistorical = [...this._bestSolution.point];

    const points = this._fireflies.map((e) => e.point);
    const fts = this._fireflies.map((e) => e.fitness);
    const binPoint = this.preprocessFunction(bestPointHistorical);
    yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: binPoint, points, fitnesses: fts };

    for (let it = 0; it < iterations; it++) {
      if (verbose) {
        console.log("it: ", it + 1, " fitness mejor: ", bestFitnessHistorical);
      }

      for (let i = 0; i < this._fireflies.length; i++) {
        for (let j = 0; j < this._fireflies.length; j++) {
          if (i === j) continue;

          if (
            (this._toMax && this._fireflies[j].fitness > this._fireflies[i].fitness) ||
            (!this._toMax && this._fireflies[j].fitness < this._fireflies[i].fitness)
          ) {
            let newPosition = this._attractAndMove(this._fireflies[i], this._fireflies[j]);
            newPosition = this.repairFunction(newPosition);
            const preprocessed = this.preprocessFunction(newPosition);
            const newFitness = this.objectiveFunction(preprocessed);
            if (newFitness !== false && newFitness !== undefined && newFitness !== null) {
              this._fireflies[i] = new SolutionBasic(newPosition, newFitness);
            }
          }
        }
      }

      if (this._fireflies.length > 0) {
        const currentBest = this.findBestSolution(this._fireflies);
        if (currentBest && this._bestSolution) {
          if (
            (this._toMax && currentBest.fitness > this._bestSolution.fitness) ||
            (!this._toMax && currentBest.fitness < this._bestSolution.fitness)
          ) {
            this._bestSolution = currentBest;
          }
        }
      }

      this._alpha *= 0.97;

      iteration++;

      if (
        (this._toMax && this._bestSolution!.fitness > bestFitnessHistorical) ||
        (!this._toMax && this._bestSolution!.fitness < bestFitnessHistorical)
      ) {
        bestFitnessHistorical = this._bestSolution!.fitness;
        bestPointHistorical = [...this._bestSolution!.point];
      }

      const pts = this._fireflies.map((e) => e.point);
      const fitnesses = this._fireflies.map((e) => e.fitness);
      const bp = this.preprocessFunction(bestPointHistorical);
      yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: bp, points: pts, fitnesses };
    }
  }

  run(
    iterations: number = 100,
    population: number = 30,
    alpha: number = 0.5,
    beta0: number = 1.0,
    gamma: number = 1.0,
    seed?: number,
    verbose: boolean = false
  ): RunResult {
    let result: RunResult = { fitness: Infinity, point: [] };
    for (const state of this.runYielded(iterations, population, alpha, beta0, gamma, seed, verbose)) {
      result = { fitness: state.bestFitness, point: state.bestBinPoint };
    }
    return result;
  }
}
