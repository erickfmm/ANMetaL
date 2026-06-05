export abstract class IProblem {
  abstract objectiveFunction(point: number[]): number | false;
  abstract preprocessFunction(point: number[]): number[];
  abstract repairFunction(point: number[]): number[];
}
