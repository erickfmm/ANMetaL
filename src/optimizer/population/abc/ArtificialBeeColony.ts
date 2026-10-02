import { IMetaheuristic, YieldResult, RunResult } from "../../IMetaheuristic";
import { SolutionBasic } from "../../ISolution";
import { SeededRandom } from "../../../utils/random";

export class ArtificialBeeColony extends IMetaheuristic {
  private _min: number;
  private _max: number;
  private _ndims: number;
  private _foodSources: SolutionBasic[] = [];
  private _trials: number[] = [];
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
    const colonySize = Math.floor(populationSize / 2);
    this._foodSources = [];
    this._trials = [];
    for (let i = 0; i < colonySize; i++) {
      const point = this._rng.randomArray(this._ndims, this._min, this._max);
      const repaired = this.repairFunction(point);
      const processed = this.preprocessFunction(repaired);
      const fitness = this.objectiveFunction(processed) as number;
      this._foodSources.push(new SolutionBasic(processed, fitness));
      this._trials.push(0);
    }
  }

  private _calculateProbability(solution: SolutionBasic): number {
    const best = this.findBestSolution(this._foodSources);
    if (this._toMax) {
      return 0.9 * (solution.fitness / best.fitness) + 0.1;
    }
    return 1.0 / (1.0 + Math.abs(solution.fitness));
  }

  private _generateNewPosition(current: number[], partner: number[]): number[] {
    const phi = this._rng.nextFloat(-1, 1);
    const newPosition = [...current];
    const dim = this._rng.nextInt(0, this._ndims);
    newPosition[dim] = newPosition[dim] + phi * (newPosition[dim] - partner[dim]);
    return IMetaheuristic.cutModPoint(newPosition, Array(this._ndims).fill(this._min), Array(this._ndims).fill(this._max));
  }

  private _employedBeePhase(): void {
    for (let i = 0; i < this._foodSources.length; i++) {
      let partner: number;
      do {
        partner = this._rng.nextInt(0, this._foodSources.length);
      } while (partner === i);

      const newPosition = this._generateNewPosition(this._foodSources[i].point, this._foodSources[partner].point);
      const repaired = this.repairFunction(newPosition);
      const processed = this.preprocessFunction(repaired);
      const newFitness = this.objectiveFunction(processed) as number;

      if (
        (this._toMax && newFitness > this._foodSources[i].fitness) ||
        (!this._toMax && newFitness < this._foodSources[i].fitness)
      ) {
        this._foodSources[i].moveTo(processed, newFitness);
        this._trials[i] = 0;
      } else {
        this._trials[i]++;
      }
    }
  }

  private _onlookerBeePhase(): void {
    const probabilities = this._foodSources.map((fs) => this._calculateProbability(fs));
    const sum = probabilities.reduce((a, b) => a + b, 0);
    const normalized = probabilities.map((p) => p / sum);

    for (let i = 0; i < this._foodSources.length; i++) {
      const selectedIdx = this._rng.choiceWeighted(normalized);

      let partner: number;
      do {
        partner = this._rng.nextInt(0, this._foodSources.length);
      } while (partner === selectedIdx);

      const newPosition = this._generateNewPosition(this._foodSources[selectedIdx].point, this._foodSources[partner].point);
      const repaired = this.repairFunction(newPosition);
      const processed = this.preprocessFunction(repaired);
      const newFitness = this.objectiveFunction(processed) as number;

      if (
        (this._toMax && newFitness > this._foodSources[selectedIdx].fitness) ||
        (!this._toMax && newFitness < this._foodSources[selectedIdx].fitness)
      ) {
        this._foodSources[selectedIdx].moveTo(processed, newFitness);
        this._trials[selectedIdx] = 0;
      } else {
        this._trials[selectedIdx]++;
      }
    }
  }

  private _scoutBeePhase(limit: number): void {
    for (let i = 0; i < this._foodSources.length; i++) {
      if (this._trials[i] >= limit) {
        const point = this._rng.randomArray(this._ndims, this._min, this._max);
        const repaired = this.repairFunction(point);
        const processed = this.preprocessFunction(repaired);
        const fitness = this.objectiveFunction(processed) as number;
        this._foodSources[i].moveTo(processed, fitness);
        this._trials[i] = 0;
      }
    }
  }

  *runYielded(
    iterations: number,
    population: number,
    limit: number = 100,
    seed: number = 0,
    verbose: boolean = false
  ): Generator<YieldResult> {
    this._iterations = iterations;
    this._rng = new SeededRandom(seed);
    this.initializePopulation(population);

    const best = this.findBestSolution(this._foodSources);
    let bestOverall = new SolutionBasic([...best.point], best.fitness);

    for (let iter = 0; iter < iterations; iter++) {
      this._employedBeePhase();
      this._onlookerBeePhase();
      this._scoutBeePhase(limit);

      const currentBest = this.findBestSolution(this._foodSources);
      if (
        (this._toMax && currentBest.fitness > bestOverall.fitness) ||
        (!this._toMax && currentBest.fitness < bestOverall.fitness)
      ) {
        bestOverall = new SolutionBasic([...currentBest.point], currentBest.fitness);
      }

      const points = this._foodSources.map((fs) => fs.point);
      const fitnesses = this._foodSources.map((fs) => fs.fitness);

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
    limit: number = 100,
    seed: number = 0,
    verbose: boolean = false
  ): RunResult {
    let bestResult: RunResult = { fitness: Infinity, point: [] };
    if (this._toMax) {
      bestResult.fitness = -Infinity;
    }
    for (const result of this.runYielded(iterations, population, limit, seed, verbose)) {
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
