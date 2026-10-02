import { YieldResult, RunResult } from "../../IMetaheuristic";
import { SolutionParticle } from "../../ISolution";
import { SeededRandom } from "../../../utils/random";
import { PSOMH_Real } from "./PSOMH_Real";

export class PSOMH_Real_WithLeap extends PSOMH_Real {
  protected moveRandom(leapPercentage: number): void {
    const numToLeap = Math.floor(this._population! * leapPercentage);
    const indexList = Array.from({ length: this._population! }, (_, i) => i);
    this._rng.shuffle(indexList);
    const indexToLeap = indexList.slice(0, numToLeap);
    const range = this._max - this._min;
    for (const solIndex of indexToLeap) {
      const [genPoint, genFitness] = this.generateRandomPoint();
      const velocity = this._rng.randomArray(this._ndims, -range, range);
      this._group[solIndex].moveTo(genPoint, genFitness);
      this._group[solIndex].setVelocity(velocity);
    }
  }

  *runYielded(
    iterations: number = 100,
    population: number = 30,
    seed: number = 42,
    omega: number = 0.5,
    phiP: number = 1,
    phiG: number = 1,
    stagnationVariation: number = 0.2,
    itsStagnation: number | null = null,
    leapPercentage: number = 0.5
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
    let tau = 1;
    let bestSolutionHistorical = this.findBestSolution(this._group);
    let bestFitnessHistorical = bestSolutionHistorical.fitness;
    let bestPointHistorical = [...bestSolutionHistorical.point];
    let fitnessAnteriorEstancado = bestFitnessHistorical;

    let points = this._group.map((e: SolutionParticle) => e.point);
    let fts = this._group.map((e: SolutionParticle) => e.fitness);
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
      points = this._group.map((e: SolutionParticle) => e.point);
      fts = this._group.map((e: SolutionParticle) => e.fitness);
      binPoint = this.preprocessFunction(bestPointHistorical);
      yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: binPoint, points, fitnesses: fts };
    }

    binPoint = this.preprocessFunction(bestPointHistorical);
    points = this._group.map((e: SolutionParticle) => e.point);
    fts = this._group.map((e: SolutionParticle) => e.fitness);
    yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: binPoint, points, fitnesses: fts };
  }

  run(
    iterations: number = 100,
    population: number = 30,
    seed: number = 42,
    omega: number = 0.5,
    phiP: number = 1,
    phiG: number = 1,
    stagnationVariation: number = 0.2,
    itsStagnation: number | null = null,
    leapPercentage: number = 0.5
  ): RunResult {
    let result!: YieldResult;
    for (result of this.runYielded(iterations, population, seed, omega, phiP, phiG, stagnationVariation, itsStagnation, leapPercentage)) {
      continue;
    }
    return { fitness: result.bestFitness, point: result.bestBinPoint };
  }
}
