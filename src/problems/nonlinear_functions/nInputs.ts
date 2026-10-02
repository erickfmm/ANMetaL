export class Brown1 {
  static func(x: number[]): number {
    let result = 0;
    for (const val of x) {
      result += val - 3;
    }
    result = result * result;
    for (let i = 0; i < x.length - 1; i++) {
      result += Math.pow(10, -3) * Math.pow(x[i] - 3, 2) - (x[i] - x[i + 1]) + Math.exp(20 * (x[i] - x[i + 1]));
    }
    return result;
  }

  static funcSimple(...args: number[]): number {
    return Brown1.func(args);
  }

  static getLimits(): { min: number; max: number } {
    return { min: -1, max: 4 };
  }

  static getTheoreticalOptimum(): number {
    return 2.0;
  }

  static getType(): string {
    return "min";
  }
}

export class Brown3 {
  static func(x: number[]): number {
    let result = 0;
    for (let i = 0; i < x.length - 1; i++) {
      result += Math.pow(x[i] * x[i], x[i + 1] * x[i + 1] + 1);
      result += Math.pow(x[i + 1] * x[i + 1], x[i] * x[i] + 1);
    }
    return result;
  }

  static funcSimple(...args: number[]): number {
    return Brown3.func(args);
  }

  static getLimits(): { min: number; max: number } {
    return { min: -1, max: 4 };
  }

  static getTheoreticalOptimum(): number {
    return 0.0;
  }

  static getType(): string {
    return "min";
  }
}

export class F10n {
  static func(x: number[]): number {
    let result = 0;
    for (let i = 0; i < x.length - 1; i++) {
      result += Math.pow(x[i] - 1, 2) * (1 + 10 * Math.pow(Math.sin(Math.PI * x[i + 1]), 2));
    }
    result += 10 * Math.pow(Math.sin(Math.PI * x[0]), 2);
    result += Math.pow(x[x.length - 1] - 1, 2);
    result = (Math.PI / x.length) * result;
    return result;
  }

  static funcSimple(...args: number[]): number {
    return F10n.func(args);
  }

  static getLimits(): { min: number; max: number } {
    return { min: -10, max: 10 };
  }

  static getTheoreticalOptimum(): number {
    return 0.0;
  }

  static getType(): string {
    return "min";
  }
}

export class F15n {
  static func(x: number[]): number {
    let result = 0;
    for (let i = 0; i < x.length - 1; i++) {
      result += Math.pow(x[i] - 1, 2) * (1 + Math.pow(Math.sin(3 * Math.PI * x[i + 1]), 2));
    }
    result += Math.pow(Math.sin(3 * Math.PI * x[0]), 2);
    result += (1 / 10.0) * Math.pow(x[x.length - 1] - 1, 2) * (1 + Math.sin(2 * Math.PI * x[x.length - 1]));
    result = (1 / 10.0) * result;
    return result;
  }

  static funcSimple(...args: number[]): number {
    return F15n.func(args);
  }

  static getLimits(): { min: number; max: number } {
    return { min: -10, max: 10 };
  }

  static getTheoreticalOptimum(): number {
    return 0.0;
  }

  static getType(): string {
    return "min";
  }
}

export class Sphere {
  static func(x: number[]): number {
    let result = 0;
    for (const val of x) {
      result += val * val;
    }
    return result;
  }

  static funcSimple(...args: number[]): number {
    return Sphere.func(args);
  }

  static getLimits(): { min: number; max: number } {
    return { min: -100, max: 100 };
  }

  static getTheoreticalOptimum(): number {
    return 0.0;
  }

  static getType(): string {
    return "min";
  }
}

export class Rosenbrock {
  static func(x: number[]): number {
    let result = 0;
    for (let i = 0; i < x.length - 1; i++) {
      result += 100.0 * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(x[i] - 1, 2);
    }
    return result;
  }

  static funcSimple(...args: number[]): number {
    return Rosenbrock.func(args);
  }

  static getLimits(): { min: number; max: number } {
    return { min: -30, max: 30 };
  }

  static getTheoreticalOptimum(): number {
    return 0.0;
  }

  static getType(): string {
    return "min";
  }
}

export class Griewank {
  static func(x: number[]): number {
    let rSum = 0;
    for (const val of x) {
      rSum += val * val;
    }
    let rProd = 1;
    for (let i = 0; i < x.length; i++) {
      rProd *= Math.cos(x[i] / Math.sqrt(i + 1));
    }
    return (1 / 4000.0) * rSum - rProd + 1;
  }

  static funcSimple(...args: number[]): number {
    return Griewank.func(args);
  }

  static getLimits(): { min: number; max: number } {
    return { min: -600, max: 600 };
  }

  static getTheoreticalOptimum(): number {
    return 0.0;
  }

  static getType(): string {
    return "min";
  }
}

export class Rastrigrin {
  static func(x: number[]): number {
    let result = 0;
    for (const val of x) {
      result += val * val - 10 * Math.cos(2 * Math.PI * val) + 10;
    }
    return result;
  }

  static funcSimple(...args: number[]): number {
    return Rastrigrin.func(args);
  }

  static getLimits(): { min: number; max: number } {
    return { min: -5.12, max: 5.12 };
  }

  static getTheoreticalOptimum(): number {
    return 0.0;
  }

  static getType(): string {
    return "min";
  }
}

export class Sumsquares {
  static func(x: number[]): number {
    let result = 0;
    for (let i = 0; i < x.length; i++) {
      result += (i + 1) * x[i] * x[i];
    }
    return result;
  }

  static funcSimple(...args: number[]): number {
    return Sumsquares.func(args);
  }

  static getLimits(): { min: number; max: number } {
    return { min: -10, max: 10 };
  }

  static getTheoreticalOptimum(): number {
    return 0.0;
  }

  static getType(): string {
    return "min";
  }
}

export class Michalewicz {
  static func(x: number[]): number {
    const m = 10;
    let result = 0;
    for (let i = 0; i < x.length; i++) {
      result += Math.sin(x[i]) * Math.pow(Math.sin(((i + 1) * x[i] * x[i]) / Math.PI), 2 * m);
    }
    return -result;
  }

  static funcSimple(...args: number[]): number {
    return Michalewicz.func(args);
  }

  static getLimits(): { min: number; max: number } {
    return { min: 0, max: Math.PI };
  }

  static getTheoreticalOptimum(): number {
    return 0.0;
  }

  static getType(): string {
    return "min";
  }
}

export class Quartic {
  static func(x: number[]): number {
    let result = 0;
    for (let i = 0; i < x.length; i++) {
      result += (i + 1) * Math.pow(x[i], 4) + (1 / (i + 2));
    }
    return result;
  }

  static funcSimple(...args: number[]): number {
    return Quartic.func(args);
  }

  static getLimits(): { min: number; max: number } {
    return { min: -1.28, max: 1.28 };
  }

  static getTheoreticalOptimum(): number {
    return 0.0;
  }

  static getType(): string {
    return "min";
  }
}

export class Schwefel {
  static func(x: number[]): number {
    let result = 0;
    for (const val of x) {
      result += val * Math.sin(Math.sqrt(Math.abs(val)));
    }
    return 418.9829 * x.length - result;
  }

  static funcSimple(...args: number[]): number {
    return Schwefel.func(args);
  }

  static getLimits(): { min: number; max: number } {
    return { min: -500, max: 500 };
  }

  static getTheoreticalOptimum(): number {
    return 0.0;
  }

  static getType(): string {
    return "min";
  }
}

export class Penalty {
  private static yi(xi: number): number {
    return 1 + (xi + 1) / 4;
  }

  private static u(xi: number, a: number, k: number, m: number): number {
    if (xi > a) {
      return k * Math.pow(xi - a, m);
    } else if (xi >= -a) {
      return 0;
    } else {
      return k * Math.pow(-xi - a, m);
    }
  }

  static func(x: number[]): number {
    let sumU = 0;
    for (const val of x) {
      sumU += Penalty.u(val, 10, 100, 4);
    }
    let result = 0;
    for (let i = 0; i < x.length - 1; i++) {
      result += Math.pow(Penalty.yi(x[i]) - 1, 2) * (1 + 10 * Math.pow(Math.sin(Math.PI * Penalty.yi(x[i + 1])), 2));
    }
    result += 10 * Math.sin(Math.PI * Penalty.yi(x[0]));
    result += Math.pow(Penalty.yi(x[x.length - 1]) + 1, 2);
    result = (Math.PI / x.length) * result;
    result += sumU;
    return result;
  }

  static funcSimple(...args: number[]): number {
    return Penalty.func(args);
  }

  static getLimits(): { min: number; max: number } {
    return { min: -50, max: 50 };
  }

  static getTheoreticalOptimum(): number {
    return 0.0;
  }

  static getType(): string {
    return "min";
  }
}
