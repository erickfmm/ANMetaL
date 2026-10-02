import { IMetaheuristic, YieldResult, RunResult } from "../../IMetaheuristic";
import { SolutionWithId } from "../../ISolution";
import { SeededRandom } from "../../../utils/random";

function argsort(arr: number[]): number[] {
  return arr
    .map((val, idx) => ({ val, idx }))
    .sort((a, b) => a.val - b.val)
    .map(({ idx }) => idx);
}

export class GeneticMH_Categorical extends IMetaheuristic {
  protected _group: SolutionWithId[] = [];
  protected _categorics: (number | string)[][];
  protected _ndims: number;
  protected _rng!: SeededRandom;
  protected _seed!: number;
  protected _mutability!: number;
  protected _fidelity!: boolean;
  protected _mutationInParents!: boolean;
  movements: Record<string, number> = {};

  constructor(
    categorics: (number | string)[][],
    ndims: number,
    toMax: boolean,
    objectiveFunction: (point: number[]) => number | false,
    repairFunction: (point: number[]) => number[],
    preprocessFunction?: (point: number[]) => number[]
  ) {
    super(toMax);
    this._categorics = categorics;
    this._ndims = ndims;
    this.objectiveFunction = objectiveFunction;
    this.repairFunction = repairFunction;
    this.preprocessFunction = preprocessFunction ?? ((p: number[]) => p);
  }

  *runYielded(
    iterations: number = 100,
    population: number = 30,
    elitistPercentage: number = 0.3,
    mutability: number = 0.1,
    fidelity: boolean = true,
    mutationInParents: boolean = true,
    seed: number = 42,
    verbose: boolean = false
  ): Generator<YieldResult> {
    this._iterations = iterations;
    this._population = population;
    this._seed = seed;
    this._mutability = mutability;
    this._fidelity = fidelity;
    this._mutationInParents = mutationInParents;
    this.movements = {
      random: 0,
      init: 0,
      select_elite: 0,
      recombination: 0,
      mutation: 0,
      crossing: 0,
    };

    this._rng = new SeededRandom(seed);
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
      if (verbose) {
        console.log("it:", iteration, "best historical fitness:", bestFitnessHistorical);
      }
      const elite = this.selectElite(elitistPercentage);
      this.recombination(elite);
      if (!this._mutationInParents) {
        for (let i = 0; i < this._group.length; i++) {
          this._group[i] = this.mutateIndividual(this._group[i]);
        }
      }
      iteration++;
      const bestSolutionIt = this.findBestSolution(this._group);
      const bestFitnessIt = bestSolutionIt.fitness;
      const bestPointIt = [...bestSolutionIt.point];
      if (
        (this._toMax && bestFitnessIt > bestFitnessHistorical) ||
        (!this._toMax && bestFitnessIt < bestFitnessHistorical)
      ) {
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
    elitistPercentage: number = 0.3,
    mutability: number = 0.1,
    fidelity: boolean = true,
    mutationInParents: boolean = true,
    seed: number = 42,
    verbose: boolean = false
  ): RunResult {
    let result!: YieldResult;
    for (result of this.runYielded(iterations, population, elitistPercentage, mutability, fidelity, mutationInParents, seed, verbose)) {
      continue;
    }
    return { fitness: result.bestFitness, point: result.bestBinPoint };
  }

  initializePopulation(population: number): void {
    this.movements["init"]++;
    this._group = [];
    for (let i = 0; i < population; i++) {
      const [point, fitness] = this.generateRandomPoint();
      this._group.push(new SolutionWithId(i, point, fitness));
    }
  }

  protected sortGroup(): void {
    const fitnesses = this._group.map((e) => e.fitness);
    const sortedIndices = argsort(fitnesses);
    this._group = sortedIndices.map((i) => this._group[i]);
  }

  protected selectElite(elitistPercentage: number): SolutionWithId[] {
    this.movements["select_elite"]++;
    this.sortGroup();
    if (Math.floor(this._group.length * elitistPercentage) <= 0) {
      throw new Error("Elitist percentage is too low");
    }
    return this._group.slice(0, Math.floor(this._group.length * elitistPercentage));
  }

  protected recombination(elite: SolutionWithId[]): void {
    this.movements["recombination"]++;
    const newGroup: SolutionWithId[] = [];
    const indexes: number[] = Array.from({ length: elite.length }, (_, i) => i);
    this._rng.shuffle(indexes);
    let males: number[] = [];
    let females: number[] = [];
    if (this._fidelity) {
      const indexesLen = Math.floor(indexes.length / 2);
      males = indexes.slice(0, indexesLen);
      females = indexes.slice(indexesLen, indexesLen + males.length);
    }
    let iParent = 0;
    for (let iIndividual = 0; iIndividual < this._population!; iIndividual++) {
      if (this._fidelity) {
        const [point, fitness] = this.crossCouple(elite[males[iParent]], elite[females[iParent]]);
        newGroup.push(new SolutionWithId(iIndividual, point, fitness));
        iParent++;
        if (iParent >= males.length) {
          iParent = 0;
        }
      } else {
        this._rng.nextInt(0, indexes.length - 1);
        const [point, fitness] = this.crossCouple(elite[indexes[iParent]], elite[indexes[iParent]]);
        newGroup.push(new SolutionWithId(iIndividual, point, fitness));
        iParent++;
        if (iParent >= indexes.length) {
          iParent = 0;
        }
      }
    }
    this._group = newGroup;
  }

  protected crossCouple(individual1: SolutionWithId, individual2: SolutionWithId): [number[], number] {
    this.movements["crossing"]++;
    let individualPoint1 = individual1.point;
    let individualPoint2 = individual2.point;
    if (individualPoint1.length !== individualPoint2.length) {
      throw new Error("Length of points are different when crossing couples");
    }
    if (this._mutationInParents) {
      individualPoint1 = this.mutateIndividual(individual1).point;
      individualPoint2 = this.mutateIndividual(individual2).point;
    }
    const mask: number[] = Array.from({ length: individualPoint1.length }, (_, i) => i % 2);
    this._rng.shuffle(mask);
    const result: number[] = [];
    for (let i = 0; i < mask.length; i++) {
      result.push(mask[i] === 0 ? individualPoint1[i] : individualPoint2[i]);
    }
    return this.repairOrNot(result);
  }

  protected mutateIndividual(individual: SolutionWithId): SolutionWithId {
    this.movements["mutation"]++;
    const individualPoint = individual.point;
    let indexesToShuffle = Array.from({ length: individualPoint.length }, (_, i) => i);
    this._rng.shuffle(indexesToShuffle);
    indexesToShuffle = indexesToShuffle.slice(0, Math.floor(indexesToShuffle.length * this._mutability));
    for (const idx of indexesToShuffle) {
      individualPoint[idx] = this._categorics[idx][this._rng.nextInt(0, this._categorics[idx].length)] as number;
    }
    const [point, fitness] = this.repairOrNot(individual.point);
    return new SolutionWithId(individual.getId(), point, fitness);
  }

  protected generateRandomPoint(): [number[], number] {
    const point: number[] = [];
    for (let i = 0; i < this._ndims; i++) {
      point.push(this._categorics[i][this._rng.nextInt(0, this._categorics[i].length)] as number);
    }
    return this.repairOrNot(point);
  }

  protected repairOrNot(point: number[]): [number[], number] {
    const fitness = this.objectiveFunction(this.preprocessFunction(point));
    if (!fitness) {
      const newPoint = this.repairFunction(point);
      const newFitness = this.objectiveFunction(this.preprocessFunction(newPoint));
      return [newPoint, newFitness as number];
    }
    return [point, fitness];
  }
}
