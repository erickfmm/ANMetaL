import { describe, it, expect } from "vitest";
import * as ANMetaL from "anmetal";

describe("Goldsteinprice", () => {
  it("func([0, -1]) should return 3.0 (the minimum)", () => {
    expect(ANMetaL.Goldsteinprice.func([0, -1])).toBeCloseTo(3.0, 10);
  });

  it("getLimits() should return correct bounds", () => {
    const limits = ANMetaL.Goldsteinprice.getLimits();
    expect(limits.min).toEqual([-2, -2]);
    expect(limits.max).toEqual([2, 2]);
  });

  it("getTheoreticalOptimum() should return 3", () => {
    expect(ANMetaL.Goldsteinprice.getTheoreticalOptimum()).toBe(3);
  });

  it("getType() should return 'min'", () => {
    expect(ANMetaL.Goldsteinprice.getType()).toBe("min");
  });
});

describe("Camelback", () => {
  it("func([0.0898, -0.7126]) should be near -1.03163", () => {
    expect(ANMetaL.Camelback.func([0.0898, -0.7126])).toBeCloseTo(-1.03163, 3);
  });

  it("getTheoreticalOptimum() should return -1.03163", () => {
    expect(ANMetaL.Camelback.getTheoreticalOptimum()).toBe(-1.03163);
  });

  it("getLimits() should return correct bounds", () => {
    const limits = ANMetaL.Camelback.getLimits();
    expect(limits.min).toEqual([-3, -2]);
    expect(limits.max).toEqual([3, 2]);
  });
});

describe("Sphere", () => {
  it("func([0, 0, 0]) should return 0", () => {
    expect(ANMetaL.Sphere.func([0, 0, 0])).toBe(0);
  });

  it("getTheoreticalOptimum() should return 0", () => {
    expect(ANMetaL.Sphere.getTheoreticalOptimum()).toBe(0);
  });

  it("func should return sum of squares", () => {
    expect(ANMetaL.Sphere.func([1, 2, 3])).toBe(14);
  });
});

describe("PartitionReal", () => {
  it("should create with seed=0 and numDims=10", () => {
    const problem = new ANMetaL.PartitionReal(0, 10);
    expect(problem.data.length).toBe(10);
    expect(problem.ndim).toBe(12);
  });

  it("objectiveFunction should return a number for a valid point", () => {
    const problem = new ANMetaL.PartitionReal(0, 10);
    const point = problem.repairFunction([5, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const processed = problem.preprocessFunction(point);
    const result = problem.objectiveFunction(processed);
    expect(typeof result).toBe("number");
    expect(Number.isFinite(result as number)).toBe(true);
  });

  it("repairFunction should return array of correct length", () => {
    const problem = new ANMetaL.PartitionReal(0, 10);
    const point = [5, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    const repaired = problem.repairFunction(point);
    expect(repaired.length).toBe(12);
  });
});

describe("KnapsackCategorical", () => {
  it("should create with seed=0", () => {
    const problem = new ANMetaL.KnapsackCategorical(50, 10, 0, 5, 5);
    expect(problem.elements.length).toBe(10);
    expect(problem.knapsackCapacity).toBe(50);
  });

  it("getPossibleCategories should return correct categories", () => {
    const problem = new ANMetaL.KnapsackCategorical(50, 10, 0, 5, 5);
    const categories = problem.getPossibleCategories();
    expect(categories.length).toBe(10);
    for (const cats of categories) {
      expect(cats).toEqual(["is", "not"]);
    }
  });

  it("objectiveFunction should return a number for a valid selection", () => {
    const problem = new ANMetaL.KnapsackCategorical(50, 10, 0, 5, 5);
    const selection = Array(10).fill("not");
    const result = problem.objectiveFunction(selection);
    expect(typeof result).toBe("number");
  });
});

describe("Sudoku", () => {
  it("should create with default empty grid", () => {
    const sudoku = new ANMetaL.Sudoku();
    expect(sudoku.state.length).toBe(9);
    for (const row of sudoku.state) {
      expect(row.length).toBe(9);
    }
  });

  it("getViolations should return a number for all-zero grid", () => {
    const sudoku = new ANMetaL.Sudoku();
    const violations = sudoku.getViolations();
    expect(typeof violations).toBe("number");
    expect(violations).toBe(0);
  });

  it("getViolations should detect row duplicates", () => {
    const state = Array.from({ length: 9 }, () => new Array(9).fill(0) as number[]);
    state[0][0] = 5;
    state[0][1] = 5;
    const sudoku = new ANMetaL.Sudoku(state);
    expect(sudoku.getViolations()).toBeGreaterThan(0);
  });

  it("isValidSolution should return true for empty grid", () => {
    const sudoku = new ANMetaL.Sudoku();
    expect(sudoku.isValidSolution()).toBe(true);
  });
});
