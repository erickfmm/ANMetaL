import { YieldResult, RunResult } from "../../IMetaheuristic";
import { SolutionWithId } from "../../ISolution";
import { SeededRandom } from "../../../utils/random";
import { GeneticMH_Categorical } from "./GeneticMH_Categorical";

export class GeneticMH_Categorical_WithLeap extends GeneticMH_Categorical {
  *runYielded(
    iterations: number = 100,
    population: number = 30,
    elitistPercentage: number = 0.3,
    mutability: number = 0.1,
    fidelity: boolean = true,
    mutationInParents: boolean = true,
    seed: number = 42,
    verbose: boolean = false,
    stagnationVariation: number = 0.2,
    itsStagnation: number = 5,
    leapPercentage: number = 0.5
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
    let tau = 1;
    let bestSolutionHistorical = this.findBestSolution(this._group);
    let bestFitnessHistorical = bestSolutionHistorical.fitness;
    let bestPointHistorical = [...bestSolutionHistorical.point];
    let fitnessAnteriorEstancado = bestFitnessHistorical;

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
      if (itsStagnation !== null && iteration === tau * itsStagnation) {
        const fitnessMejorActual = this.findBestSolution(this._group).fitness;
        const variation = Math.abs(fitnessAnteriorEstancado - fitnessMejorActual) / fitnessAnteriorEstancado;
        if (variation < stagnationVariation) {
          this.moveRandom(leapPercentage);
        }
        fitnessAnteriorEstancado = fitnessMejorActual;
        tau++;
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
    verbose: boolean = false,
    stagnationVariation: number = 0.2,
    itsStagnation: number = 5,
    leapPercentage: number = 0.5
  ): RunResult {
    let result!: YieldResult;
    for (result of this.runYielded(iterations, population, elitistPercentage, mutability, fidelity, mutationInParents, seed, verbose, stagnationVariation, itsStagnation, leapPercentage)) {
      continue;
    }
    return { fitness: result.bestFitness, point: result.bestBinPoint };
  }

  protected moveRandom(leapPercentage: number): void {
    this.movements["random"]++;
    let indexes = Array.from({ length: this._group.length }, (_, i) => i);
    this._rng.shuffle(indexes);
    indexes = indexes.slice(0, Math.floor(indexes.length * leapPercentage));
    for (const i of indexes) {
      this._group.splice(i, 1);
      const [point, fitness] = this.generateRandomPoint();
      this._group.push(new SolutionWithId(i, point, fitness));
    }
  }
}
