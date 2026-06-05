export class SeededRandom {
  private state: number;

  constructor(seed: number) {
    this.state = seed;
  }

  next(): number {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  nextInt(min: number, max: number): number {
    return Math.floor(this.next() * (max - min)) + min;
  }

  nextFloat(min: number = 0, max: number = 1): number {
    return this.next() * (max - min) + min;
  }

  shuffle<T>(array: T[]): T[] {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      const tmp = array[i];
      array[i] = array[j];
      array[j] = tmp;
    }
    return array;
  }

  choice<T>(array: T[]): T {
    return array[Math.floor(this.next() * array.length)];
  }

  randomArray(length: number, min: number = 0, max: number = 1): number[] {
    const result: number[] = new Array(length);
    for (let i = 0; i < length; i++) {
      result[i] = this.nextFloat(min, max);
    }
    return result;
  }

  normal(mean: number = 0, std: number = 1): number {
    const u1 = this.next();
    const u2 = this.next();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return z0 * std + mean;
  }

  normalArray(length: number, mean: number = 0, std: number = 1): number[] {
    const result: number[] = new Array(length);
    for (let i = 0; i < length; i++) {
      result[i] = this.normal(mean, std);
    }
    return result;
  }

  choiceWeighted(probabilities: number[]): number {
    const r = this.next();
    let cumulative = 0;
    for (let i = 0; i < probabilities.length; i++) {
      cumulative += probabilities[i];
      if (r <= cumulative) return i;
    }
    return probabilities.length - 1;
  }
}
