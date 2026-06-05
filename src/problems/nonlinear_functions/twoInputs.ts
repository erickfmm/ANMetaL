export class Camelback {
  static func(x: number[]): number {
    const xv = x[0];
    const yv = x[1];
    let result = (4 - 2.1 * xv * xv + (Math.pow(xv, 4)) / 3.0) * (xv * xv);
    result += xv * yv;
    result += (4 * yv * yv - 4) * (yv * yv);
    return result;
  }

  static funcSimple(...args: number[]): number {
    return Camelback.func(args);
  }

  static getLimits(): { min: number[]; max: number[] } {
    return { min: [-3, -2], max: [3, 2] };
  }

  static getTheoreticalOptimum(): number {
    return -1.03163;
  }

  static getType(): string {
    return "min";
  }
}

export class Goldsteinprice {
  static func(x: number[]): number {
    const xv = x[0];
    const yv = x[1];
    let result = 1 + Math.pow(xv + yv + 1, 2) * (19 - 14 * xv + 3 * xv * xv - 14 * yv + 6 * xv * yv + 3 * yv * yv);
    result *= (30 + Math.pow(2 * xv - 3 * yv, 2) * (18 - 32 * xv + 12 * xv * xv + 48 * yv - 36 * xv * yv + 27 * yv * yv));
    return result;
  }

  static funcSimple(...args: number[]): number {
    return Goldsteinprice.func(args);
  }

  static getLimits(): { min: number[]; max: number[] } {
    return { min: [-2, -2], max: [2, 2] };
  }

  static getTheoreticalOptimum(): number {
    return 3;
  }

  static getType(): string {
    return "min";
  }
}

export class Pshubert1 {
  static func(x: number[]): number {
    const xv = x[0];
    const yv = x[1];
    let res1 = 0;
    for (let i = 1; i <= 5; i++) {
      res1 += i * Math.cos((i + 1) * xv + i);
    }
    let res2 = 0;
    for (let i = 1; i <= 5; i++) {
      res2 += i * Math.cos((i + 1) * yv + i);
    }
    let result = 0.5 * (Math.pow(xv + 1.42513, 2) + Math.pow(yv + 0.80032, 2));
    result = res1 * res2 + result;
    return result;
  }

  static funcSimple(...args: number[]): number {
    return Pshubert1.func(args);
  }

  static getLimits(): { min: number[]; max: number[] } {
    return { min: [-10, -10], max: [10, 10] };
  }

  static getTheoreticalOptimum(): number {
    return -186.73091;
  }

  static getType(): string {
    return "min";
  }
}

export class Pshubert2 {
  static func(x: number[]): number {
    const xv = x[0];
    const yv = x[1];
    let res1 = 0;
    for (let i = 1; i <= 5; i++) {
      res1 += i * Math.cos((i + 1) * xv + i);
    }
    let res2 = 0;
    for (let i = 1; i <= 5; i++) {
      res2 += i * Math.cos((i + 1) * yv + i);
    }
    let result = Math.pow(xv + 1.42513, 2) + Math.pow(yv + 0.80032, 2);
    result = res1 * res2 + result;
    return result;
  }

  static funcSimple(...args: number[]): number {
    return Pshubert2.func(args);
  }

  static getLimits(): { min: number[]; max: number[] } {
    return { min: [-10, -10], max: [10, 10] };
  }

  static getTheoreticalOptimum(): number {
    return -186.73091;
  }

  static getType(): string {
    return "min";
  }
}

export class Shubert {
  static func(x: number[]): number {
    const xv = x[0];
    const yv = x[1];
    let res1 = 0;
    for (let i = 1; i <= 5; i++) {
      res1 += i * Math.cos((i + 1) * xv + i);
    }
    let res2 = 0;
    for (let i = 1; i <= 5; i++) {
      res2 += i * Math.cos((i + 1) * yv + i);
    }
    return res1 * res2;
  }

  static funcSimple(...args: number[]): number {
    return Shubert.func(args);
  }

  static getLimits(): { min: number[]; max: number[] } {
    return { min: [-10, -10], max: [10, 10] };
  }

  static getTheoreticalOptimum(): number {
    return -186.73091;
  }

  static getType(): string {
    return "min";
  }
}

export class Quartic {
  static func(x: number[]): number {
    const xv = x[0];
    const yv = x[1];
    return Math.pow(xv, 4) / 4 - (xv * xv) / 2 + xv / 10 + (yv * yv) / 2;
  }

  static funcSimple(...args: number[]): number {
    return Quartic.func(args);
  }

  static getLimits(): { min: number[]; max: number[] } {
    return { min: [-10, -10], max: [10, 10] };
  }

  static getTheoreticalOptimum(): number {
    return -0.35239;
  }

  static getType(): string {
    return "min";
  }
}
