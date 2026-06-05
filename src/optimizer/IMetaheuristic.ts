import { SolutionBasic } from "./ISolution";

export type YieldResult = {
  iteration: number;
  bestFitness: number;
  bestBinPoint: number[];
  points: number[][];
  fitnesses: number[];
};

export type RunResult = {
  fitness: number;
  point: number[];
};

export abstract class IMetaheuristic {
  protected _iterations: number | null = null;
  protected _population: number | null = null;
  protected _toMax: boolean;
  protected objectiveFunction!: (point: number[]) => number | false;
  protected repairFunction!: (point: number[]) => number[];
  protected preprocessFunction!: (point: number[]) => number[];

  constructor(toMax: boolean) {
    this._toMax = toMax;
  }

  abstract run(): RunResult;
  abstract initializePopulation(populationSize: number): void;

  isPointValid(point: number[]): boolean {
    const fitness = this.objectiveFunction(point);
    if (!fitness) {
      return false;
    }
    return true;
  }

  findBestPoint(points: number[][]): number[] {
    let best: number[] | null = null;
    let bestFitness: number | null = null;
    for (const point of points) {
      const fitness = this.objectiveFunction(point) as number;
      if (best === null) {
        best = point;
      }
      if (bestFitness === null) {
        bestFitness = fitness;
      }
      if (this._toMax && fitness > bestFitness!) {
        best = point;
        bestFitness = fitness;
      }
      if (!this._toMax && fitness < bestFitness!) {
        best = point;
        bestFitness = fitness;
      }
    }
    return best!;
  }

  findBestSolution(solutions: SolutionBasic[]): SolutionBasic {
    let best: SolutionBasic | null = null;
    let bestFitness: number | null = null;
    for (const sol of solutions) {
      if (best === null) {
        best = sol;
      }
      if (bestFitness === null) {
        bestFitness = sol.fitness;
      }
      if (this._toMax && sol.fitness > bestFitness!) {
        best = sol;
        bestFitness = sol.fitness;
      }
      if (!this._toMax && sol.fitness < bestFitness!) {
        best = sol;
        bestFitness = sol.fitness;
      }
    }
    return best!;
  }

  static cutModPoint(point: number[], minX: number[], maxX: number[]): number[] {
    const newPoint: number[] = [];
    for (let i = 0; i < point.length; i++) {
      const min = minX[i];
      const max = maxX[i];
      const ran = Math.abs(max - min);
      const x = point[i];
      if (x < min) {
        const dist = Math.abs(x - min);
        newPoint.push(max - (dist % ran));
      } else if (x > max) {
        const dist = Math.abs(max - x);
        newPoint.push(min + (dist % ran));
      } else {
        newPoint.push(x);
      }
    }
    return newPoint;
  }
}
