import { IMetaheuristic, YieldResult, RunResult } from "../../IMetaheuristic";
import { SolutionBasic } from "../../ISolution";
import { SeededRandom } from "../../../utils/random";

function gammaLn(z: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - gammaLn(1 - z);
  }
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) {
    x += c[i] / (z + i);
  }
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

export class CuckooSearch extends IMetaheuristic {
  private _min: number;
  private _max: number;
  private _ndims: number;
  private _rng!: SeededRandom;
  private _nests: SolutionBasic[] = [];
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
    this._nests = [];
    for (let i = 0; i < populationSize; i++) {
      const point = this._rng.randomArray(this._ndims, this._min, this._max);
      const repaired = this.repairFunction(point);
      const preprocessed = this.preprocessFunction(repaired);
      const fitness = this.objectiveFunction(preprocessed);
      if (fitness !== false && fitness !== undefined && fitness !== null) {
        this._nests.push(new SolutionBasic(point, fitness));
      }
    }
    while (this._nests.length < 1) {
      const point = this._rng.randomArray(this._ndims, this._min, this._max);
      const repaired = this.repairFunction(point);
      const preprocessed = this.preprocessFunction(repaired);
      const fitness = this.objectiveFunction(preprocessed);
      if (fitness !== false && fitness !== undefined && fitness !== null) {
        this._nests.push(new SolutionBasic(point, fitness));
      }
    }
    if (this._nests.length > 0) {
      this._bestSolution = this.findBestSolution(this._nests);
    }
  }

  private _levyFlight(): number[] {
    const beta = 3 / 2;
    const sigmaU = Math.exp(
      gammaLn(1 + beta) +
        Math.log(Math.sin((Math.PI * beta) / 2)) -
        gammaLn((1 + beta) / 2) -
        Math.log(beta) -
        ((beta - 1) / 2) * Math.log(2)
    );
    const sigma = Math.pow(sigmaU, 1 / beta);
    const u = this._rng.normalArray(this._ndims, 0, sigma);
    const v = this._rng.normalArray(this._ndims, 0, 1);
    const step: number[] = new Array(this._ndims);
    for (let i = 0; i < this._ndims; i++) {
      step[i] = u[i] / Math.pow(Math.abs(v[i]), 1 / beta);
    }
    return step;
  }

  private _getCuckoo(currentNest: SolutionBasic): number[] {
    const stepSize = 0.01;
    const step = this._levyFlight();
    const newPosition: number[] = new Array(this._ndims);
    for (let i = 0; i < this._ndims; i++) {
      const pos = currentNest.point[i] + stepSize * step[i];
      newPosition[i] = Math.max(this._min, Math.min(this._max, pos));
    }
    return newPosition;
  }

  private _abandonNests(pa: number): void {
    for (let i = 0; i < this._nests.length; i++) {
      if (this._rng.next() < pa) {
        const newPoint = this._rng.randomArray(this._ndims, this._min, this._max);
        const repaired = this.repairFunction(newPoint);
        const preprocessed = this.preprocessFunction(repaired);
        const fitness = this.objectiveFunction(preprocessed);
        if (fitness !== false && fitness !== undefined && fitness !== null) {
          this._nests[i] = new SolutionBasic(newPoint, fitness);
        }
      }
    }
  }

  *runYielded(
    iterations: number = 100,
    population: number = 30,
    pa: number = 0.25,
    seed?: number,
    verbose: boolean = false
  ): Generator<YieldResult> {
    this._iterations = iterations;
    this._rng = new SeededRandom(seed ?? 0);

    this.initializePopulation(population);

    if (!this._nests || !this._bestSolution) {
      return;
    }

    let iteration = 1;
    let bestFitnessHistorical = this._bestSolution.fitness;
    let bestPointHistorical = [...this._bestSolution.point];

    const points = this._nests.map((e) => e.point);
    const fts = this._nests.map((e) => e.fitness);
    const binPoint = this.preprocessFunction(bestPointHistorical);
    yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: binPoint, points, fitnesses: fts };

    for (let it = 0; it < iterations; it++) {
      if (verbose) {
        console.log("it: ", it + 1, " fitness mejor: ", bestFitnessHistorical);
      }

      const i = this._rng.nextInt(0, this._nests.length);
      const cuckooNest = this._nests[i];

      let newPosition = this._getCuckoo(cuckooNest);
      newPosition = this.repairFunction(newPosition);
      newPosition = this.preprocessFunction(newPosition);
      const newFitness = this.objectiveFunction(newPosition);

      if (newFitness !== false && newFitness !== undefined && newFitness !== null) {
        const j = this._rng.nextInt(0, this._nests.length);
        if (
          (this._toMax && newFitness > this._nests[j].fitness) ||
          (!this._toMax && newFitness < this._nests[j].fitness)
        ) {
          this._nests[j] = new SolutionBasic(newPosition, newFitness);
        }
      }

      this._abandonNests(pa);

      if (this._nests.length > 0) {
        const currentBest = this.findBestSolution(this._nests);
        if (currentBest && this._bestSolution) {
          if (
            (this._toMax && currentBest.fitness > this._bestSolution.fitness) ||
            (!this._toMax && currentBest.fitness < this._bestSolution.fitness)
          ) {
            this._bestSolution = currentBest;
          }
        }
      }

      iteration++;

      if (
        (this._toMax && this._bestSolution!.fitness > bestFitnessHistorical) ||
        (!this._toMax && this._bestSolution!.fitness < bestFitnessHistorical)
      ) {
        bestFitnessHistorical = this._bestSolution!.fitness;
        bestPointHistorical = [...this._bestSolution!.point];
      }

      const pts = this._nests.map((e) => e.point);
      const fitnesses = this._nests.map((e) => e.fitness);
      const bp = this.preprocessFunction(bestPointHistorical);
      yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: bp, points: pts, fitnesses };
    }
  }

  run(
    iterations: number = 100,
    population: number = 30,
    pa: number = 0.25,
    seed?: number,
    verbose: boolean = false
  ): RunResult {
    let result: RunResult = { fitness: Infinity, point: [] };
    for (const state of this.runYielded(iterations, population, pa, seed, verbose)) {
      result = { fitness: state.bestFitness, point: state.bestBinPoint };
    }
    return result;
  }
}
