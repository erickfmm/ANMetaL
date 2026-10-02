import { YieldResult, RunResult } from "../../IMetaheuristic";
import { SolutionWithId } from "../../ISolution";
import { SeededRandom } from "../../../utils/random";
import { GreedyMH_Real } from "./GreedyMH_Real";

export class GreedyMH_Real_WithLeap extends GreedyMH_Real {
  protected moveRandom(leapPercentage: number): void {
    const numToLeap = Math.floor(this._population! * leapPercentage);
    const indexList = Array.from({ length: this._population! }, (_, i) => i);
    this._rng.shuffle(indexList);
    const indexToLeap = indexList.slice(0, numToLeap);
    for (const solIndex of indexToLeap) {
      const [genPoint, genFitness] = this.generateRandomPoint();
      this._group[solIndex].moveTo(genPoint, genFitness);
    }
  }

  *runYielded(
    iterations: number = 100,
    population: number = 30,
    seed: number = 42,
    stagnationVariation: number = 0.2,
    itsStagnation: number = 5,
    leapPercentage: number = 0.5
  ): Generator<YieldResult> {
    this._iterations = iterations;
    this._population = population;
    this._seed = seed;
    this._rng = new SeededRandom(seed);

    this.initializePopulation(population);
    let iteration = 1;
    let tau = 1;
    let bestSolutionHistorical = this.findBestSolution(this._group);
    let bestFitnessHistorical = bestSolutionHistorical.fitness;
    let bestPointHistorical = [...bestSolutionHistorical.point];
    let fitnessAnteriorEstancado = bestFitnessHistorical;

    let points = this._group.map((e: SolutionWithId) => e.point);
    let fts = this._group.map((e: SolutionWithId) => e.fitness);
    let binPoint = this.preprocessFunction(bestPointHistorical);
    yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: binPoint, points, fitnesses: fts };

    while (iteration <= iterations) {
      for (const individual of this._group) {
        const [resultPoint, fitness] = this.move(individual);
        individual.moveTo(resultPoint, fitness);
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
      if ((this._toMax && bestFitnessIt > bestFitnessHistorical) ||
          (!this._toMax && bestFitnessIt < bestFitnessHistorical)) {
        bestFitnessHistorical = bestFitnessIt;
        bestPointHistorical = bestPointIt;
      }
      points = this._group.map((e: SolutionWithId) => e.point);
      fts = this._group.map((e: SolutionWithId) => e.fitness);
      binPoint = this.preprocessFunction(bestPointHistorical);
      yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: binPoint, points, fitnesses: fts };
    }

    binPoint = this.preprocessFunction(bestPointHistorical);
    points = this._group.map((e: SolutionWithId) => e.point);
    fts = this._group.map((e: SolutionWithId) => e.fitness);
    yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: binPoint, points, fitnesses: fts };
  }

  run(
    iterations: number = 100,
    population: number = 30,
    seed: number = 42,
    stagnationVariation: number = 0.2,
    itsStagnation: number = 5,
    leapPercentage: number = 0.5
  ): RunResult {
    let result!: YieldResult;
    for (result of this.runYielded(iterations, population, seed, stagnationVariation, itsStagnation, leapPercentage)) {
      continue;
    }
    return { fitness: result.bestFitness, point: result.bestBinPoint };
  }
}
