import { IMetaheuristic, YieldResult, RunResult } from "../../IMetaheuristic";
import { SolutionBasic } from "../../ISolution";
import { SeededRandom } from "../../../utils/random";

export class HarmonySearch extends IMetaheuristic {
  private _min: number;
  private _max: number;
  private _ndims: number;
  private _rng!: SeededRandom;
  private _hms: number = 30;
  private _hmcr: number = 0.9;
  private _par: number = 0.3;
  private _bw: number = 0.01;
  private _harmonyMemory: SolutionBasic[] = [];
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
    this._hms = populationSize;
    this._harmonyMemory = [];
    for (let i = 0; i < this._hms; i++) {
      const point = this._rng.randomArray(this._ndims, this._min, this._max);
      const repaired = this.repairFunction(point);
      const preprocessed = this.preprocessFunction(repaired);
      const fitness = this.objectiveFunction(preprocessed);
      if (fitness !== false && fitness !== undefined && fitness !== null) {
        this._harmonyMemory.push(new SolutionBasic(point, fitness));
      }
    }
    if (this._harmonyMemory.length > 0) {
      this._bestSolution = this.findBestSolution(this._harmonyMemory);
    }
  }

  private _memoryConsideration(dimension: number): number {
    const randomHarmony = this._rng.choice(this._harmonyMemory);
    return randomHarmony.point[dimension];
  }

  private _pitchAdjustment(value: number): number {
    if (this._rng.next() < this._par) {
      return value + this._bw * this._rng.nextFloat(-1, 1);
    }
    return value;
  }

  private _createNewHarmony(): number[] {
    const newHarmony: number[] = new Array(this._ndims);
    for (let i = 0; i < this._ndims; i++) {
      if (this._rng.next() < this._hmcr) {
        let value = this._memoryConsideration(i);
        value = this._pitchAdjustment(value);
        newHarmony[i] = value;
      } else {
        newHarmony[i] = this._rng.nextFloat(this._min, this._max);
      }
    }
    for (let i = 0; i < this._ndims; i++) {
      newHarmony[i] = Math.max(this._min, Math.min(this._max, newHarmony[i]));
    }
    return newHarmony;
  }

  private _updateHarmonyMemory(newHarmony: number[], newFitness: number): void {
    let worstIndex = 0;
    let worstFitness = this._harmonyMemory[0].fitness;

    for (let i = 1; i < this._harmonyMemory.length; i++) {
      if (
        (this._toMax && this._harmonyMemory[i].fitness < worstFitness) ||
        (!this._toMax && this._harmonyMemory[i].fitness > worstFitness)
      ) {
        worstIndex = i;
        worstFitness = this._harmonyMemory[i].fitness;
      }
    }

    if (
      (this._toMax && newFitness > worstFitness) ||
      (!this._toMax && newFitness < worstFitness)
    ) {
      this._harmonyMemory[worstIndex] = new SolutionBasic(newHarmony, newFitness);
    }
  }

  *runYielded(
    iterations: number = 100,
    population: number = 30,
    hmcr: number = 0.9,
    par: number = 0.3,
    bw: number = 0.2,
    seed?: number,
    verbose: boolean = false
  ): Generator<YieldResult> {
    this._iterations = iterations;
    this._hmcr = hmcr;
    this._par = par;
    this._bw = bw;
    this._rng = new SeededRandom(seed ?? 0);

    this.initializePopulation(population);

    if (!this._harmonyMemory) {
      return;
    }

    this._bestSolution = this.findBestSolution(this._harmonyMemory);
    if (!this._bestSolution) {
      return;
    }

    let iteration = 1;
    let bestFitnessHistorical = this._bestSolution.fitness;
    let bestPointHistorical = [...this._bestSolution.point];

    const points = this._harmonyMemory.map((e) => e.point);
    const fts = this._harmonyMemory.map((e) => e.fitness);
    const binPoint = this.preprocessFunction(bestPointHistorical);
    yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: binPoint, points, fitnesses: fts };

    for (let it = 0; it < iterations; it++) {
      if (verbose) {
        console.log("it: ", it + 1, " fitness mejor: ", bestFitnessHistorical);
      }

      const newHarmony = this._createNewHarmony();
      const repaired = this.repairFunction(newHarmony);
      const preprocessed = this.preprocessFunction(repaired);
      const newFitness = this.objectiveFunction(preprocessed);

      if (newFitness !== false && newFitness !== undefined && newFitness !== null) {
        this._updateHarmonyMemory(newHarmony, newFitness);

        if (this._harmonyMemory.length > 0) {
          const currentBest = this.findBestSolution(this._harmonyMemory);
          if (currentBest && this._bestSolution) {
            if (
              (this._toMax && currentBest.fitness > this._bestSolution.fitness) ||
              (!this._toMax && currentBest.fitness < this._bestSolution.fitness)
            ) {
              this._bestSolution = currentBest;
            }
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

      const pts = this._harmonyMemory.map((e) => e.point);
      const fitnesses = this._harmonyMemory.map((e) => e.fitness);
      const bp = this.preprocessFunction(bestPointHistorical);
      yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: bp, points: pts, fitnesses };
    }
  }

  run(
    iterations: number = 100,
    population: number = 30,
    hmcr: number = 0.9,
    par: number = 0.3,
    bw: number = 0.2,
    seed?: number,
    verbose: boolean = false
  ): RunResult {
    let result: RunResult = { fitness: Infinity, point: [] };
    for (const state of this.runYielded(iterations, population, hmcr, par, bw, seed, verbose)) {
      result = { fitness: state.bestFitness, point: state.bestBinPoint };
    }
    return result;
  }
}
