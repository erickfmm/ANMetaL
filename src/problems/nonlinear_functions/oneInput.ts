export class F1 {
  static func(x: number[]): number {
    const xv = x[0];
    let result = 2 * Math.pow(xv - 0.75, 2);
    result += Math.sin(5 * Math.PI * xv - 0.4 * Math.PI);
    result -= 0.125;
    return result;
  }

  static funcSimple(...args: number[]): number {
    return F1.func(args);
  }

  static getLimits(): { min: number; max: number } {
    return { min: 0, max: 1 };
  }

  static getTheoreticalOptimum(): number {
    return -1.12323;
  }

  static getType(): string {
    return "min";
  }
}

export class F3 {
  static func(x: number[]): number {
    let result = 0;
    for (let j = 1; j <= 5; j++) {
      result += j * Math.sin((j + 1) * x[0] + j);
    }
    return -1 * result;
  }

  static funcSimple(...args: number[]): number {
    return F3.func(args);
  }

  static getLimits(): { min: number; max: number } {
    return { min: -10, max: 10 };
  }

  static getTheoreticalOptimum(): number {
    return -12.03125;
  }

  static getType(): string {
    return "min";
  }
}
