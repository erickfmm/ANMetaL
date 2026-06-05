import { describe, it, expect } from "vitest";
import * as ANMetaL from "anmetal";

describe("eulerMethod", () => {
  it("approximates e for dy/dx=y with y(0)=1", () => {
    const result = ANMetaL.eulerMethod(
      (_x, y) => y,
      0,
      1,
      1,
      1000,
    );
    expect(result).toBeCloseTo(Math.E, 1);
    expect(Math.abs(result - Math.E)).toBeLessThan(0.01);
  });
});

describe("newtonMethod", () => {
  it("finds root of x^2 - 4 = 0 starting at x=3", () => {
    const root = ANMetaL.newtonMethod(
      (x) => x * x - 4,
      (x) => 2 * x,
      3,
      20,
    );
    expect(root).toBeCloseTo(2.0, 3);
    expect(Math.abs(root - 2.0)).toBeLessThan(0.001);
  });

  it("works with multiplicity m=1 (explicit)", () => {
    const root = ANMetaL.newtonMethod(
      (x) => x * x - 4,
      (x) => 2 * x,
      3,
      10,
      1,
    );
    expect(root).toBeCloseTo(2.0, 3);
  });
});
