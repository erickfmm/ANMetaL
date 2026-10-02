import { IMetaheuristic, YieldResult, RunResult } from "../../IMetaheuristic";
import { SolutionBasic } from "../../ISolution";
import { SeededRandom } from "../../../utils/random";

export class BlackHole extends IMetaheuristic {
  private _min: number;
  private _max: number;
  private _ndims: number;
  private _stars: SolutionBasic[] = [];
  private _blackHole: SolutionBasic | null = null;
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
    this._stars = [];
    for (let i = 0; i < populationSize; i++) {
      const point = this._rng.randomArray(this._ndims, this._min, this._max);
      const repaired = this.repairFunction(point);
      const processed = this.preprocessFunction(repaired);
      const fitness = this.objectiveFunction(processed) as number;
      this._stars.push(new SolutionBasic(processed, fitness));
    }
    this._blackHole = this.findBestSolution(this._stars);
  }

  private _calculateEventHorizon(): number {
    const fitnessSum = this._stars.reduce((acc, star) => acc + Math.abs(star.fitness), 0);
    if (fitnessSum === 0) return 0;
    return Math.abs(this._blackHole!.fitness) / fitnessSum;
  }

  private _moveStarTowardsBlackHole(starPosition: number[]): number[] {
    const newPosition = [...starPosition];
    for (let i = 0; i < this._ndims; i++) {
      const r = this._rng.next();
      newPosition[i] = newPosition[i] + r * (this._blackHole!.point[i] - newPosition[i]);
    }
    return IMetaheuristic.cutModPoint(
      newPosition,
      Array(this._ndims).fill(this._min),
      Array(this._ndims).fill(this._max)
    );
  }

  private _isWithinEventHorizon(position: number[], eventHorizon: number): boolean {
    let distance = 0;
    for (let i = 0; i < this._ndims; i++) {
      distance += Math.pow(position[i] - this._blackHole!.point[i], 2);
    }
    distance = Math.sqrt(distance);
    return distance < eventHorizon;
  }

  *runYielded(
    iterations: number,
    population: number,
    seed: number = 0,
    verbose: boolean = false
  ): Generator<YieldResult> {
    this._iterations = iterations;
    this._rng = new SeededRandom(seed);
    this.initializePopulation(population);

    for (let iter = 0; iter < iterations; iter++) {
      for (let i = 0; i < this._stars.length; i++) {
        const newPosition = this._moveStarTowardsBlackHole(this._stars[i].point);
        const repaired = this.repairFunction(newPosition);
        const processed = this.preprocessFunction(repaired);
        const fitness = this.objectiveFunction(processed) as number;
        this._stars[i].moveTo(processed, fitness);
      }

      const eventHorizon = this._calculateEventHorizon();

      for (let i = this._stars.length - 1; i >= 0; i--) {
        if (this._isWithinEventHorizon(this._stars[i].point, eventHorizon)) {
          const newPoint = this._rng.randomArray(this._ndims, this._min, this._max);
          const repaired = this.repairFunction(newPoint);
          const processed = this.preprocessFunction(repaired);
          const fitness = this.objectiveFunction(processed) as number;
          this._stars[i] = new SolutionBasic(processed, fitness);
        }
      }

      for (const star of this._stars) {
        if (
          (this._toMax && star.fitness > this._blackHole!.fitness) ||
          (!this._toMax && star.fitness < this._blackHole!.fitness)
        ) {
          this._blackHole = new SolutionBasic([...star.point], star.fitness);
        }
      }

      const points = this._stars.map((s) => s.point);
      const fitnesses = this._stars.map((s) => s.fitness);

      yield {
        iteration: iter,
        bestFitness: this._blackHole!.fitness,
        bestBinPoint: this._blackHole!.point,
        points,
        fitnesses,
      };
    }
  }

  run(
    iterations: number = 100,
    population: number = 50,
    seed: number = 0,
    verbose: boolean = false
  ): RunResult {
    let bestResult: RunResult = { fitness: Infinity, point: [] };
    if (this._toMax) {
      bestResult.fitness = -Infinity;
    }
    for (const result of this.runYielded(iterations, population, seed, verbose)) {
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
