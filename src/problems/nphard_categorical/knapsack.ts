import { SeededRandom } from "../../utils/random";

export interface KnapsackElement {
  value: number;
  cost: number;
}

export class KnapsackCategorical {
  knapsackCapacity: number;
  totalPossibleElements: number;
  elements: KnapsackElement[];

  constructor(
    knapsackCapacity: number,
    totalPossibleElements: number,
    seed: number,
    maxCost: number,
    maxValue: number,
    elements?: KnapsackElement[]
  ) {
    this.knapsackCapacity = knapsackCapacity;
    if (elements !== undefined && Array.isArray(elements)) {
      this.elements = elements;
      this.totalPossibleElements = elements.length;
    } else {
      this.totalPossibleElements = totalPossibleElements;
      const rnd = new SeededRandom(seed);
      this.elements = Array.from({ length: totalPossibleElements }, () => ({
        value: rnd.nextFloat(0, maxValue),
        cost: rnd.nextFloat(0, maxCost),
      }));
    }
  }

  getPossibleCategories(): string[][] {
    return Array.from({ length: this.totalPossibleElements }, () => [
      "is",
      "not",
    ]);
  }

  objectiveFunction(point: string[]): number | false {
    if (!this.isValid(point)) return false;
    let total = 0;
    for (let i = 0; i < point.length; i++) {
      if (point[i] === "is") total += this.elements[i].value;
    }
    return total;
  }

  preprocessFunction(point: string[]): string[] {
    return point;
  }

  repairFunction(point: string[]): string[] {
    let i = -1;
    while (!this.isValid(point)) {
      let found = false;
      while (!found) {
        i++;
        if (point[i] === "is") {
          point[i] = "not";
          found = true;
        }
      }
    }
    return point;
  }

  isValid(point: string[]): boolean {
    let totalCost = 0;
    for (let i = 0; i < point.length; i++) {
      if (point[i] === "is") totalCost += this.elements[i].cost;
    }
    return totalCost <= this.knapsackCapacity;
  }
}
