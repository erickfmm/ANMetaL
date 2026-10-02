import { IProblem } from "../IProblem";
import { SeededRandom } from "../../utils/random";

export abstract class PartitionSubsetAbstract extends IProblem {
  data: number[];
  ndim: number;
  min_x: number;
  max_x: number;

  constructor(seed: number, numDims: number, data?: number[]) {
    super();
    if (data !== undefined && Array.isArray(data)) {
      this.data = data;
    } else {
      const rnd = new SeededRandom(seed);
      this.data = Array.from({ length: numDims }, () =>
        rnd.nextFloat(0, numDims - 1)
      );
    }
    this.ndim = this.data.length + 2;
    this.min_x = 0;
    this.max_x = this.data.length;
  }

  isValid(point: number[]): boolean {
    const counts = new Array(this.data.length).fill(0) as number[];
    for (let i = 2; i < point.length; i++) {
      const idx = Math.floor(point[i]);
      if (idx < 0 || idx >= this.data.length) return false;
      counts[idx]++;
      if (counts[idx] > 1) return false;
    }
    for (const c of counts) {
      if (c === 0 || c > 1) return false;
    }
    return true;
  }

  repairFunction(point: number[]): number[] {
    point = point.map(Math.abs);
    const counts = new Array(this.data.length).fill(0) as number[];
    const dups: number[][] = Array.from({ length: this.data.length }, () => []);

    for (let i = 2; i < point.length; i++) {
      let idx = Math.floor(point[i]);
      if (idx < 0 || idx >= this.data.length) {
        point[i] = 0;
        idx = 0;
      }
      counts[idx]++;
      dups[idx].push(i);
    }

    const missing: number[] = [];
    for (let i = 0; i < counts.length; i++) {
      if (counts[i] === 0) missing.push(i);
    }

    const sumDups: number[] = [];
    for (let i = 0; i < dups.length; i++) {
      sumDups.push(dups[i].length);
      if (dups[i].length > 0) dups[i].splice(0, 1);
    }

    const flatDups = dups.flat();

    if (missing.length !== flatDups.length) {
      throw new Error(
        `Repair mismatch: missing=${missing.length}, dups=${flatDups.length}`
      );
    }

    for (let i = 0; i < flatDups.length; i++) {
      point[flatDups[i]] = missing[i];
    }

    return point;
  }

  preprocessFunction(point: number[]): number[] {
    const result = point.map(Math.abs);
    for (let i = 2; i < result.length; i++) {
      result[i] = Math.floor(result[i]);
    }
    return result;
  }

  getSums(point: number[]): [number | false, number | false] {
    if (!this.isValid(point)) return [false, false];
    const s1Raw = point[0];
    const s2Raw = point[1];
    let s1 = Math.floor((s1Raw / (s1Raw + s2Raw)) * this.data.length);
    s1 = s1 >= 0 && s1 < this.data.length ? s1 : 0;
    let sum1 = 0;
    let sum2 = 0;
    for (let i = 2; i < s1 + 2; i++) {
      sum1 += this.data[Math.floor(point[i])];
    }
    for (let i = s1 + 2; i < point.length; i++) {
      sum2 += this.data[Math.floor(point[i])];
    }
    return [sum1, sum2];
  }

  abstract objectiveFunction(point: number[]): number | false;
}

export class PartitionReal extends PartitionSubsetAbstract {
  objectiveFunction(point: number[]): number | false {
    const [sum1, sum2] = this.getSums(point);
    if (sum1 === false || sum2 === false) return false;
    return Math.abs(sum1 - sum2);
  }
}

export class SubsetReal extends PartitionSubsetAbstract {
  objectiveFunction(point: number[]): number | false {
    const [sum1] = this.getSums(point);
    if (sum1 === false) return false;
    return Math.abs(sum1);
  }
}
