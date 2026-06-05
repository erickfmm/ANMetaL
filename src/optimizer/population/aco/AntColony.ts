import { IMetaheuristic, YieldResult, RunResult } from "../../IMetaheuristic";
import { SolutionBasic } from "../../ISolution";
import { SeededRandom } from "../../../utils/random";

export class AntColony extends IMetaheuristic {
  private _min: number;
  private _max: number;
  private _ndims: number;
  private _discretizationPoints: number = 100;
  private _pheromone: number[][] = [];
  private _ants: SolutionBasic[] = [];
  private _rng!: SeededRandom;

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
    this.preprocessFunction = preprocessFunction ?? ((p: number[]) => p);
  }

  initializePopulation(populationSize: number): void {
    this._population = populationSize;
    this._ants = [];
    this._pheromone = [];
    for (let d = 0; d < this._ndims; d++) {
      this._pheromone[d] = new Array(this._discretizationPoints).fill(0.1);
    }
    for (let i = 0; i < populationSize; i++) {
      const point = this._rng.randomArray(this._ndims, this._min, this._max);
      const repaired = this.repairFunction(point);
      const processed = this.preprocessFunction(repaired);
      const fitness = this.objectiveFunction(processed) as number;
      this._ants.push(new SolutionBasic(processed, fitness));
    }
  }

  private _discretizePosition(position: number, dim: number): number {
    const normalized = (position - this._min) / (this._max - this._min);
    const index = Math.floor(normalized * (this._discretizationPoints - 1));
    return Math.max(0, Math.min(this._discretizationPoints - 1, index));
  }

  private _continuousPosition(index: number, dim: number): number {
    const normalized = index / (this._discretizationPoints - 1);
    return this._min + normalized * (this._max - this._min);
  }

  private _constructSolution(alpha: number, beta: number): number[] {
    const point: number[] = [];
    for (let d = 0; d < this._ndims; d++) {
      const probabilities: number[] = [];
      for (let j = 0; j < this._discretizationPoints; j++) {
        const tau = Math.pow(this._pheromone[d][j], alpha);
        const eta = Math.pow(1.0 / (Math.abs(j - this._discretizationPoints / 2) + 1), beta);
        probabilities.push(tau * eta);
      }
      const sum = probabilities.reduce((a, b) => a + b, 0);
      const normalized = probabilities.map((p) => p / sum);
      const idx = this._rng.choiceWeighted(normalized);
      point.push(this._continuousPosition(idx, d));
    }
    return point;
  }

  private _updatePheromones(evaporationRate: number): void {
    const best = this.findBestSolution(this._ants);
    for (let d = 0; d < this._ndims; d++) {
      for (let j = 0; j < this._discretizationPoints; j++) {
        this._pheromone[d][j] *= (1 - evaporationRate);
      }
    }
    for (const ant of this._ants) {
      for (let d = 0; d < this._ndims; d++) {
        const idx = this._discretizePosition(ant.point[d], d);
        const contribution = 1.0 / (1.0 + Math.abs(ant.fitness));
        this._pheromone[d][idx] += contribution;
      }
    }
    for (let d = 0; d < this._ndims; d++) {
      const idx = this._discretizePosition(best.point[d], d);
      const bestContribution = 1.0 / (1.0 + Math.abs(best.fitness));
      this._pheromone[d][idx] += bestContribution * 2;
    }
  }

  *runYielded(
    iterations: number,
    population: number,
    evaporation_rate: number = 0.1,
    alpha: number = 1.0,
    beta: number = 2.0,
    seed: number = 0,
    verbose: boolean = false
  ): Generator<YieldResult> {
    this._iterations = iterations;
    this._rng = new SeededRandom(seed);
    this.initializePopulation(population);

    const best = this.findBestSolution(this._ants);
    let bestOverall = new SolutionBasic([...best.point], best.fitness);

    for (let iter = 0; iter < iterations; iter++) {
      for (let i = 0; i < this._ants.length; i++) {
        const newPoint = this._constructSolution(alpha, beta);
        const repaired = this.repairFunction(newPoint);
        const processed = this.preprocessFunction(repaired);
        const fitness = this.objectiveFunction(processed) as number;
        this._ants[i].moveTo(processed, fitness);
      }

      this._updatePheromones(evaporation_rate);

      const currentBest = this.findBestSolution(this._ants);
      if (
        (this._toMax && currentBest.fitness > bestOverall.fitness) ||
        (!this._toMax && currentBest.fitness < bestOverall.fitness)
      ) {
        bestOverall = new SolutionBasic([...currentBest.point], currentBest.fitness);
      }

      const points = this._ants.map((a) => a.point);
      const fitnesses = this._ants.map((a) => a.fitness);

      yield {
        iteration: iter,
        bestFitness: bestOverall.fitness,
        bestBinPoint: bestOverall.point,
        points,
        fitnesses,
      };
    }
  }

  run(
    iterations: number = 100,
    population: number = 50,
    evaporation_rate: number = 0.1,
    alpha: number = 1.0,
    beta: number = 2.0,
    seed: number = 0,
    verbose: boolean = false
  ): RunResult {
    let bestResult: RunResult = { fitness: Infinity, point: [] };
    if (this._toMax) {
      bestResult.fitness = -Infinity;
    }
    for (const result of this.runYielded(iterations, population, evaporation_rate, alpha, beta, seed, verbose)) {
      if (
        (this._toMax && result.bestFitness > bestResult.fitness) ||
        (!this._toMax && result.bestFitness < bestResult.fitness)
      ) {
        bestResult = { fitness: result.bestFitness, point: result.bestBinPoint };
      }
    }
    return bestResult;
  }
}
