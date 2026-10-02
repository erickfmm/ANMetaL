import { IMetaheuristic, YieldResult, RunResult } from "../../IMetaheuristic";
import { SolutionBasic } from "../../ISolution";
import { SeededRandom } from "../../../utils/random";

export class BatAlgorithm extends IMetaheuristic {
  private _min: number;
  private _max: number;
  private _ndims: number;
  private _bats: SolutionBasic[] = [];
  private _velocities: number[][] = [];
  private _frequencies: number[] = [];
  private _pulseRates: number[] = [];
  private _loudness: number[] = [];
  private _rng!: SeededRandom;
  private _alpha: number = 0.9;
  private _gamma: number = 0.9;

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
    this._bats = [];
    this._velocities = [];
    this._frequencies = [];
    this._pulseRates = [];
    this._loudness = [];
    for (let i = 0; i < populationSize; i++) {
      const point = this._rng.randomArray(this._ndims, this._min, this._max);
      const repaired = this.repairFunction(point);
      const processed = this.preprocessFunction(repaired);
      const fitness = this.objectiveFunction(processed) as number;
      this._bats.push(new SolutionBasic(processed, fitness));
      this._velocities.push(new Array(this._ndims).fill(0));
      this._frequencies.push(0);
      this._pulseRates.push(0);
      this._loudness.push(1);
    }
  }

  private _localSearch(bat: SolutionBasic, loudness: number): number[] {
    const epsilon = this._rng.normalArray(this._ndims, 0, 1);
    const avgLoudness = this._loudness.reduce((a, b) => a + Math.abs(b), 0) / this._loudness.length;
    const newPoint = [...bat.point];
    for (let i = 0; i < this._ndims; i++) {
      newPoint[i] += epsilon[i] * avgLoudness;
    }
    return IMetaheuristic.cutModPoint(newPoint, Array(this._ndims).fill(this._min), Array(this._ndims).fill(this._max));
  }

  *runYielded(
    iterations: number,
    population: number,
    fmin: number = 0,
    fmax: number = 2,
    A: number = 0.5,
    r0: number = 0.5,
    seed: number = 0,
    verbose: boolean = false
  ): Generator<YieldResult> {
    this._iterations = iterations;
    this._rng = new SeededRandom(seed);
    this.initializePopulation(population);

    for (let i = 0; i < this._bats.length; i++) {
      this._loudness[i] = A;
      this._pulseRates[i] = r0;
    }

    const best = this.findBestSolution(this._bats);
    let bestOverall = new SolutionBasic([...best.point], best.fitness);

    for (let iter = 0; iter < iterations; iter++) {
      for (let i = 0; i < this._bats.length; i++) {
        this._frequencies[i] = fmin + (fmax - fmin) * this._rng.next();

        for (let d = 0; d < this._ndims; d++) {
          this._velocities[i][d] =
            this._velocities[i][d] +
            (this._bats[i].point[d] - bestOverall.point[d]) * this._frequencies[i];
        }

        let newPosition = [...this._bats[i].point];
        for (let d = 0; d < this._ndims; d++) {
          newPosition[d] += this._velocities[i][d];
        }
        newPosition = IMetaheuristic.cutModPoint(
          newPosition,
          Array(this._ndims).fill(this._min),
          Array(this._ndims).fill(this._max)
        );

        if (this._rng.next() > this._pulseRates[i]) {
          newPosition = this._localSearch(this._bats[i], this._loudness[i]);
        }

        const repaired = this.repairFunction(newPosition);
        const processed = this.preprocessFunction(repaired);
        const newFitness = this.objectiveFunction(processed) as number;

        if (
          this._rng.next() < this._loudness[i] &&
          ((this._toMax && newFitness > this._bats[i].fitness) ||
            (!this._toMax && newFitness < this._bats[i].fitness))
        ) {
          this._bats[i].moveTo(processed, newFitness);
          this._pulseRates[i] = r0 * (1 - Math.exp(-this._gamma * iter));
          this._loudness[i] *= this._alpha;
        }

        if (
          (this._toMax && newFitness > bestOverall.fitness) ||
          (!this._toMax && newFitness < bestOverall.fitness)
        ) {
          bestOverall = new SolutionBasic([...processed], newFitness);
        }
      }

      const points = this._bats.map((b) => b.point);
      const fitnesses = this._bats.map((b) => b.fitness);

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
    fmin: number = 0,
    fmax: number = 2,
    A: number = 0.5,
    r0: number = 0.5,
    seed: number = 0,
    verbose: boolean = false
  ): RunResult {
    let bestResult: RunResult = { fitness: Infinity, point: [] };
    if (this._toMax) {
      bestResult.fitness = -Infinity;
    }
    for (const result of this.runYielded(iterations, population, fmin, fmax, A, r0, seed, verbose)) {
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
