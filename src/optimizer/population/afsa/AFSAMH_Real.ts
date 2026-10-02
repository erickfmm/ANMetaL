import { IMetaheuristic, YieldResult, RunResult } from "../../IMetaheuristic";
import { SolutionWithId } from "../../ISolution";
import { SeededRandom } from "../../../utils/random";
import { nsphereToCartesian, distance } from "../../../utils/pointsUtils";

type MoveCounter = { move: number; prey: number; follow: number; swarm: number; leap: number };

export class AFSAMH_Real extends IMetaheuristic {
  private _swarm: SolutionWithId[] = [];
  private _rng!: SeededRandom;
  private _min: number;
  private _max: number;
  private _ndims: number;
  private _visualDistance!: number;
  private _nPointsToChoose!: number;
  private _velocityPercentage!: number;
  private _seed: number | null = null;
  private _minArr!: number[];
  private _maxArr!: number[];
  vecesMovimiento: MoveCounter = { move: 0, prey: 0, follow: 0, swarm: 0, leap: 0 };

  constructor(
    minValue: number,
    maxValue: number,
    ndims: number,
    toMax: boolean,
    objectiveFunction: (point: number[]) => number | false,
    repairFunction: (point: number[]) => number[],
    preprocessFunction: ((point: number[]) => number[]) | null = null
  ) {
    super(toMax);
    this._min = minValue;
    this._max = maxValue;
    this._ndims = ndims;
    this.objectiveFunction = objectiveFunction;
    this.repairFunction = repairFunction;
    this.preprocessFunction = preprocessFunction ?? ((p: number[]) => p);
  }

  *runYielded(
    iterations: number = 100,
    population: number = 30,
    verbose: boolean = false,
    stagnationVariation: number = 0.2,
    itsStagnation: number = 5,
    leapPercentage: number = 0.5,
    velocityPercentage: number = 0.3,
    nPointsToChoose: number = 1,
    crowdedPercentage: number = 0.9,
    visualDistancePercentage: number = 0.1,
    seed: number | null = null
  ): Generator<YieldResult> {
    this._iterations = iterations;
    this._population = population;
    this._seed = seed;
    this._visualDistance = Math.sqrt(this._max * this._max * this._ndims) * visualDistancePercentage;
    this._nPointsToChoose = nPointsToChoose;
    this._velocityPercentage = velocityPercentage;
    this._rng = new SeededRandom(seed ?? 0);
    this._minArr = new Array(this._ndims).fill(this._min);
    this._maxArr = new Array(this._ndims).fill(this._max);

    this.initializePopulation(population);
    let iteration = 1;
    let tau = 1;
    const bestSolutionHistorical = this.findBestSolution(this._swarm);
    let bestFitnessHistorical = bestSolutionHistorical.fitness;
    let bestPointHistorical = [...bestSolutionHistorical.point];
    let fitnessAnteriorEstancado = bestFitnessHistorical;

    let points = this._swarm.map((e) => e.point);
    let fts = this._swarm.map((e) => e.fitness);
    let binPoint = this.preprocessFunction(bestPointHistorical);
    yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: binPoint, points, fitnesses: fts };

    while (iteration <= iterations) {
      if (verbose) {
        console.log("it: ", iteration, " fitness mejor: ", bestFitnessHistorical);
      }
      for (const fish of this._swarm) {
        const { neighborhood } = this.getNeighborhood(fish);
        let resultPoint: number[];
        if (neighborhood.length === 0) {
          resultPoint = this.AFMove(fish);
        } else {
          if (neighborhood.length / this._swarm.length >= crowdedPercentage) {
            resultPoint = this.AFFollow(fish, neighborhood);
          } else {
            const resultPoint1 = this.AFSwarm(fish, neighborhood);
            const resultPoint2 = this.AFPrey(fish, neighborhood);
            resultPoint = this.findBestPoint([resultPoint1, resultPoint2]);
          }
        }
        const [repairedPoint, fitness] = this.repairOrNot(resultPoint);
        fish.moveTo(repairedPoint, fitness);
      }

      if (iteration === tau * itsStagnation) {
        const fitnessMejorActual = this.findBestSolution(this._swarm).fitness;
        let variation = 0;
        if (fitnessAnteriorEstancado !== 0) {
          variation = Math.abs(fitnessAnteriorEstancado - fitnessMejorActual) / fitnessAnteriorEstancado;
        }
        if (variation < stagnationVariation) {
          this.AFLeap(leapPercentage);
        }
        fitnessAnteriorEstancado = fitnessMejorActual;
        tau += 1;
      }
      iteration += 1;

      const bestSolutionIt = this.findBestSolution(this._swarm);
      const bestFitnessIt = bestSolutionIt.fitness;
      const bestPointIt = [...bestSolutionIt.point];
      if (this._toMax && bestFitnessIt > bestFitnessHistorical) {
        bestFitnessHistorical = bestFitnessIt;
        bestPointHistorical = bestPointIt;
      }
      if (!this._toMax && bestFitnessIt < bestFitnessHistorical) {
        bestFitnessHistorical = bestFitnessIt;
        bestPointHistorical = bestPointIt;
      }

      points = this._swarm.map((e) => e.point);
      fts = this._swarm.map((e) => e.fitness);
      binPoint = this.preprocessFunction(bestPointHistorical);
      yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: binPoint, points, fitnesses: fts };
    }

    points = this._swarm.map((e) => e.point);
    fts = this._swarm.map((e) => e.fitness);
    binPoint = this.preprocessFunction(bestPointHistorical);
    yield { iteration, bestFitness: bestFitnessHistorical, bestBinPoint: binPoint, points, fitnesses: fts };
  }

  run(
    iterations: number = 100,
    population: number = 30,
    verbose: boolean = false,
    stagnationVariation: number = 0.2,
    itsStagnation: number = 5,
    leapPercentage: number = 0.5,
    velocityPercentage: number = 0.3,
    nPointsToChoose: number = 1,
    crowdedPercentage: number = 0.9,
    visualDistancePercentage: number = 0.1,
    seed: number | null = null
  ): RunResult {
    let bestFitness: number = 0;
    let bestBinPoint: number[] = [];
    for (const result of this.runYielded(
      iterations,
      population,
      verbose,
      stagnationVariation,
      itsStagnation,
      leapPercentage,
      velocityPercentage,
      nPointsToChoose,
      crowdedPercentage,
      visualDistancePercentage,
      seed
    )) {
      bestFitness = result.bestFitness;
      bestBinPoint = result.bestBinPoint;
    }
    return { fitness: bestFitness, point: bestBinPoint };
  }

  initializePopulation(populationSize: number): void {
    this._swarm = [];
    for (let i = 0; i < populationSize; i++) {
      const [point, fitness] = this.generateRandomPoint();
      const fish = new SolutionWithId(i, point, fitness);
      this._swarm.push(fish);
    }
  }

  private AFMove(fish: SolutionWithId): number[] {
    this.vecesMovimiento.move += 1;
    const points: number[][] = [];
    for (let i = 0; i < this._nPointsToChoose; i++) {
      points.push(this.generateInVisual(fish.point));
    }
    return this.findBestPoint(points);
  }

  private AFSwarm(fish: SolutionWithId, neighborhood: SolutionWithId[]): number[] {
    this.vecesMovimiento.swarm += 1;
    const centralPoint = this.getCentralPoint(neighborhood);
    if (
      this.isPointValid(centralPoint) &&
      this.findBestPoint([centralPoint, fish.point]) === centralPoint
    ) {
      return centralPoint;
    } else {
      return this.AFFollow(fish, neighborhood);
    }
  }

  private AFPrey(fish: SolutionWithId, neighborhood: SolutionWithId[]): number[] {
    this.vecesMovimiento.prey += 1;
    const bestNeighbour = this.findBestSolution(neighborhood) as SolutionWithId;
    if (this.findBestSolution([bestNeighbour, fish]) === fish) {
      return this.getCloserTo(fish.point, bestNeighbour.point);
    } else {
      return this.AFFollow(fish, neighborhood);
    }
  }

  private AFFollow(fish: SolutionWithId, neighborhood: SolutionWithId[]): number[] {
    this.vecesMovimiento.follow += 1;
    const points: number[][] = [];
    for (let i = 0; i < this._nPointsToChoose; i++) {
      const neighborIdx = Math.floor(this._rng.nextFloat(0, neighborhood.length - 1));
      points.push(neighborhood[neighborIdx].point);
    }
    return this.getCloserTo(fish.point, this.findBestPoint(points));
  }

  private AFLeap(leapPercentage: number): void {
    this.vecesMovimiento.leap += 1;
    const numFishesToLeap = Math.floor(this._population! * leapPercentage);
    const indexFishesToLeap = Array.from({ length: this._population! }, (_, i) => i);
    this._rng.shuffle(indexFishesToLeap);
    const selected = indexFishesToLeap.slice(0, numFishesToLeap);
    for (const fishIndex of selected) {
      const [genPoint, genFitness] = this.generateRandomPoint();
      this._swarm[fishIndex].moveTo(genPoint, genFitness);
    }
  }

  private getNeighborhood(fish: SolutionWithId): { neighborhood: SolutionWithId[]; neighborhoodPoints: number[][] } {
    const neighborhood: SolutionWithId[] = [];
    const neighborhoodPoints: number[][] = [];
    for (const f of this._swarm) {
      if (f.getId() !== fish.getId()) {
        const d = distance(fish.point, f.point);
        if (d <= this._visualDistance) {
          neighborhood.push(f);
          neighborhoodPoints.push(f.point);
        }
      }
    }
    return { neighborhood, neighborhoodPoints };
  }

  private getCentralPoint(neighborhood: SolutionWithId[]): number[] {
    const mediaDim = new Array(this._ndims).fill(0);
    for (const neighbor of neighborhood) {
      const p = neighbor.point;
      for (let i = 0; i < mediaDim.length; i++) {
        mediaDim[i] += p[i];
      }
    }
    for (let i = 0; i < mediaDim.length; i++) {
      mediaDim[i] /= neighborhood.length;
    }
    return this.repairOrNot(mediaDim)[0];
  }

  private generateInVisual(originPoint: number[]): number[] {
    const dist = this._rng.nextFloat(0, this._visualDistance);
    const angles = this._rng.randomArray(this._ndims - 1, 0, 2 * Math.PI);
    const cartesianPoint = nsphereToCartesian(dist, angles);
    for (let i = 0; i < cartesianPoint.length; i++) {
      cartesianPoint[i] += originPoint[i];
    }
    return this.repairOrNot(cartesianPoint)[0];
  }

  protected generateRandomPoint(): [number[], number] {
    const cartesianPoint = this._rng.randomArray(this._ndims, this._min, this._max);
    return this.repairOrNot(cartesianPoint);
  }

  protected repairOrNot(cartesianPoint: number[]): [number[], number] {
    let point = IMetaheuristic.cutModPoint(cartesianPoint, this._minArr, this._maxArr);
    const fitness = this.objectiveFunction(this.preprocessFunction(point));
    if (!fitness) {
      const newPoint = this.repairFunction(point);
      const newFitness = this.objectiveFunction(this.preprocessFunction(newPoint)) as number;
      return [newPoint, newFitness];
    } else {
      return [point, fitness];
    }
  }

  private getCloserTo(originPoint: number[], destPoint: number[]): number[] {
    const origin = [...originPoint];
    for (let i = 0; i < origin.length; i++) {
      origin[i] += this._velocityPercentage * (origin[i] - destPoint[i]);
    }
    return this.repairOrNot(origin)[0];
  }
}
