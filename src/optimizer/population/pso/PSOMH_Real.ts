import { IMetaheuristic, YieldResult, RunResult } from "../../IMetaheuristic";
import { SolutionParticle } from "../../ISolution";
import { SeededRandom } from "../../../utils/random";

export class PSOMH_Real extends IMetaheuristic {
  protected _group: SolutionParticle[] = [];
  protected _rng!: SeededRandom;
  protected _min: number;
  protected _max: number;
  protected _ndims: number;
  protected _seed!: number;
  protected omega!: number;
  protected phiP!: number;
  protected phiG!: number;

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

  initializePopulation(population: number): void {
    this._group = [];
    for (let i = 0; i < population; i++) {
      const [point, fitness] = this.generateRandomPoint();
      const range = this._max - this._min;
      const velocity = this._rng.randomArray(this._ndims, -range, range);
      this._group.push(new SolutionParticle(i, point, fitness, velocity));
    }
  }

  *runYielded(
    iterations: number = 100,
    population: number = 30,
    seed: number = 42,
    omega: number = 0.5,
    phiP: number = 1,
    phiG: number = 1
  ): Generator<YieldResult> {
    this._iterations = iterations;
    this._population = population;
    this._seed = seed;
    this._rng = new SeededRandom(seed);
    this.omega = omega;
    this.phiP = phiP;
    this.phiG = phiG;

    this.initializePopulation(population);
    let iteration = 1;
    let bestSolutionHistorical = this.findBestSolution(this._group);
    let bestFitnessHistorical = bestSolutionHistorical.fitness;
    let bestPointHistorical = [...bestSolutionHistorical.point];

    let points = this._group.map((e) => e.point);
    let fts = this._group.map((e) => e.fitness);
    let binPoint = this.preprocessFunction(bestPointHistorical);
    yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: binPoint, points, fitnesses: fts };

    while (iteration <= iterations) {
      for (const individual of this._group) {
        for (let idim = 0; idim < individual.point.length; idim++) {
          const rp = this._rng.next();
          const rg = this._rng.next();
          individual.velocity[idim] =
            omega * individual.velocity[idim] +
            phiP * rp * (individual.bestPoint[idim] - individual.point[idim]) +
            phiG * rg * (bestPointHistorical[idim] - individual.point[idim]);
        }
        const [resultPoint, fitness] = this.move(individual);
        individual.moveTo(resultPoint, fitness);
        if ((this._toMax && individual.fitness > individual.bestFitness) ||
            (!this._toMax && individual.fitness < individual.bestFitness)) {
          individual.setBestPoint(resultPoint, fitness);
        }
      }
      iteration++;
      const bestSolutionIt = this.findBestSolution(this._group);
      const bestFitnessIt = bestSolutionIt.fitness;
      const bestPointIt = [...bestSolutionIt.point];
      if ((this._toMax && bestFitnessIt > bestFitnessHistorical) ||
          (!this._toMax && bestFitnessIt < bestFitnessHistorical)) {
        bestFitnessHistorical = bestFitnessIt;
        bestPointHistorical = bestPointIt;
      }
      points = this._group.map((e) => e.point);
      fts = this._group.map((e) => e.fitness);
      binPoint = this.preprocessFunction(bestPointHistorical);
      yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: binPoint, points, fitnesses: fts };
    }

    binPoint = this.preprocessFunction(bestPointHistorical);
    points = this._group.map((e) => e.point);
    fts = this._group.map((e) => e.fitness);
    yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: binPoint, points, fitnesses: fts };
  }

  run(
    iterations: number = 100,
    population: number = 30,
    seed: number = 42,
    omega: number = 0.5,
    phiP: number = 1,
    phiG: number = 1
  ): RunResult {
    let result!: YieldResult;
    for (result of this.runYielded(iterations, population, seed, omega, phiP, phiG)) {
      continue;
    }
    return { fitness: result.bestFitness, point: result.bestBinPoint };
  }

  protected move(individual: SolutionParticle): [number[], number] {
    const originPoint = [...individual.point];
    for (let idim = 0; idim < originPoint.length; idim++) {
      originPoint[idim] += individual.velocity[idim];
    }
    return this.repairOrNot(originPoint);
  }

  protected generateRandomPoint(): [number[], number] {
    const cartesianPoint = this._rng.randomArray(this._ndims, this._min, this._max);
    return this.repairOrNot(cartesianPoint);
  }

  protected repairOrNot(cartesianPoint: number[]): [number[], number] {
    const range = this._max - this._min;
    const minX = new Array(this._ndims).fill(this._min) as number[];
    const maxX = new Array(this._ndims).fill(this._max) as number[];
    let point = IMetaheuristic.cutModPoint(cartesianPoint, minX, maxX);
    let fitness = this.objectiveFunction(this.preprocessFunction(point));
    if (!fitness) {
      const newPoint = this.repairFunction(point);
      fitness = this.objectiveFunction(this.preprocessFunction(newPoint));
      return [newPoint, fitness as number];
    }
    return [point, fitness];
  }
}
