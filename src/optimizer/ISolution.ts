export class SolutionBasic {
  point: number[];
  fitness: number;

  constructor(point: number[], fitness: number) {
    this.point = point;
    this.fitness = fitness;
  }

  moveTo(newPoint: number[], newFitness: number): void {
    this.point = newPoint;
    this.fitness = newFitness;
  }
}

export class SolutionWithId extends SolutionBasic {
  protected _id: number;

  constructor(id: number, point: number[], fitness: number) {
    super(point, fitness);
    this._id = id;
  }

  setId(id: number): void {
    this._id = id;
  }

  getId(): number {
    return this._id;
  }
}

export class SolutionParticle extends SolutionWithId {
  velocity: number[];
  bestPoint: number[];
  bestFitness: number;

  constructor(id: number, point: number[], fitness: number, velocity: number[]) {
    super(id, point, fitness);
    this.velocity = velocity;
    this.bestPoint = [...point];
    this.bestFitness = fitness;
  }

  setBestPoint(newPoint: number[], newFitness: number): void {
    this.bestPoint = newPoint;
    this.bestFitness = newFitness;
  }

  setVelocity(velocity: number[]): void {
    this.velocity = velocity;
  }

  getVelocity(): number[] {
    return this.velocity;
  }
}
