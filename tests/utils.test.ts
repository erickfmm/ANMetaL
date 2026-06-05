import { describe, it, expect } from "vitest";
import * as ANMetaL from "anmetal";

describe("SeededRandom", () => {
  it("next() returns reproducible values", () => {
    const rng1 = new ANMetaL.SeededRandom(42);
    const rng2 = new ANMetaL.SeededRandom(42);
    const vals1 = [rng1.next(), rng1.next(), rng1.next()];
    const vals2 = [rng2.next(), rng2.next(), rng2.next()];
    expect(vals1).toEqual(vals2);
  });

  it("nextInt() returns integers in range", () => {
    const rng = new ANMetaL.SeededRandom(42);
    for (let i = 0; i < 100; i++) {
      const val = rng.nextInt(0, 10);
      expect(Number.isInteger(val)).toBe(true);
      expect(val).toBeGreaterThanOrEqual(0);
      expect(val).toBeLessThan(10);
    }
  });

  it("shuffle() shuffles deterministically", () => {
    const rng1 = new ANMetaL.SeededRandom(42);
    const rng2 = new ANMetaL.SeededRandom(42);
    const arr1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const arr2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    rng1.shuffle(arr1);
    rng2.shuffle(arr2);
    expect(arr1).toEqual(arr2);
    expect(arr1).not.toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  });

  it("nextFloat() returns values in range", () => {
    const rng = new ANMetaL.SeededRandom(42);
    for (let i = 0; i < 100; i++) {
      const val = rng.nextFloat(5, 10);
      expect(val).toBeGreaterThanOrEqual(5);
      expect(val).toBeLessThan(10);
    }
  });

  it("normal() returns a number", () => {
    const rng = new ANMetaL.SeededRandom(42);
    const val = rng.normal();
    expect(typeof val).toBe("number");
    expect(Number.isFinite(val)).toBe(true);
  });

  it("choiceWeighted() selects based on probabilities", () => {
    const rng = new ANMetaL.SeededRandom(42);
    const counts = [0, 0, 0];
    for (let i = 0; i < 10000; i++) {
      const idx = rng.choiceWeighted([0.1, 0.3, 0.6]);
      counts[idx]++;
    }
    expect(counts[2]).toBeGreaterThan(counts[1]);
    expect(counts[1]).toBeGreaterThan(counts[0]);
  });
});

describe("pointsUtils", () => {
  it("distance() computes Euclidean distance (3-4-5 triangle)", () => {
    expect(ANMetaL.distance([0, 0], [3, 4])).toBe(5);
  });

  it("distance() returns 0 for identical points", () => {
    expect(ANMetaL.distance([1, 1, 1], [1, 1, 1])).toBe(0);
  });

  it("distance() throws for mismatched dimensions", () => {
    expect(() => ANMetaL.distance([0, 0], [1, 2, 3])).toThrow();
  });

  it("distanceSquared() computes squared Euclidean distance", () => {
    expect(ANMetaL.distanceSquared([0, 0], [3, 4])).toBe(25);
  });

  it("distanceSquared() throws for mismatched dimensions", () => {
    expect(() => ANMetaL.distanceSquared([0, 0], [1, 2, 3])).toThrow();
  });

  it("distanceTaxicab() computes Manhattan distance", () => {
    expect(ANMetaL.distanceTaxicab([0, 0], [3, 4])).toBe(7);
  });

  it("distanceTaxicab() throws for mismatched dimensions", () => {
    expect(() => ANMetaL.distanceTaxicab([0, 0], [1, 2, 3])).toThrow();
  });

  it("nsphereToCartesian() returns correct cartesian for unit circle angle=0", () => {
    const result = ANMetaL.nsphereToCartesian(1, [0]);
    expect(result[0]).toBeCloseTo(1, 10);
    expect(result[1]).toBeCloseTo(0, 10);
  });
});

describe("binarizationFloat", () => {
  it("sShape1(0) returns 0.5", () => {
    expect(ANMetaL.sShape1(0)).toBeCloseTo(0.5, 10);
  });

  it("vShape1(0) returns 0", () => {
    expect(ANMetaL.vShape1(0)).toBeCloseTo(0, 6);
  });

  const floatFns = [
    ["sShape1", ANMetaL.sShape1],
    ["sShape2", ANMetaL.sShape2],
    ["sShape3", ANMetaL.sShape3],
    ["sShape4", ANMetaL.sShape4],
    ["vShape1", ANMetaL.vShape1],
    ["vShape2", ANMetaL.vShape2],
    ["vShape3", ANMetaL.vShape3],
    ["vShape4", ANMetaL.vShape4],
  ] as const;

  for (const [name, fn] of floatFns) {
    it(`${name}() returns values in [0,1] for inputs in [-5, 5]`, () => {
      for (let x = -5; x <= 5; x += 0.5) {
        const val = fn(x);
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThanOrEqual(1);
      }
    });
  }
});

describe("binarizationStrategy", () => {
  it("standard() returns 1 when uniformValue <= x", () => {
    expect(ANMetaL.standard(0.7, 0.5)).toBe(1);
  });

  it("standard() returns 0 when uniformValue > x", () => {
    expect(ANMetaL.standard(0.3, 0.5)).toBe(0);
  });

  it("elitist() returns 1 when u1 < x and u2 <= x", () => {
    expect(ANMetaL.elitist(0.7, 0.5, 0.6)).toBe(1);
  });

  it("elitist() returns 0 when u1 < x and u2 > x", () => {
    expect(ANMetaL.elitist(0.7, 0.5, 0.8)).toBe(0);
  });

  it("elitist() returns 0 when u1 >= x", () => {
    expect(ANMetaL.elitist(0.4, 0.5, 0.1)).toBe(0);
  });
});
