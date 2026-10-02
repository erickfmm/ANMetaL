import { describe, it, expect } from "vitest";
import * as ANMetaL from "anmetal";

const objFunc = (p: number[]) => ANMetaL.Goldsteinprice.func(p);
const repair = (p: number[]) => p;
const preprocess = (p: number[]) => p;
const limits = ANMetaL.Goldsteinprice.getLimits();

function expectValidResult(result: ANMetaL.RunResult, ndims: number) {
  expect(result).toHaveProperty("fitness");
  expect(result).toHaveProperty("point");
  expect(typeof result.fitness).toBe("number");
  expect(Number.isFinite(result.fitness)).toBe(true);
  expect(Array.isArray(result.point)).toBe(true);
  expect(result.point.length).toBe(ndims);
}

describe("PSO", () => {
  it("should initialize without throwing", () => {
    expect(() => new ANMetaL.PSOMH_Real(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess)).not.toThrow();
  });

  it("should return a valid result from run()", () => {
    const pso = new ANMetaL.PSOMH_Real(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess);
    const result = pso.run(10, 15, 42, 0.8, 0.5, 0.5);
    expectValidResult(result, 2);
  });
});

describe("PSOWL", () => {
  it("should initialize without throwing", () => {
    expect(() => new ANMetaL.PSOMH_Real_WithLeap(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess)).not.toThrow();
  });

  it("should return a valid result from run()", () => {
    const pso = new ANMetaL.PSOMH_Real_WithLeap(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess);
    const result = pso.run(10, 15, 42, 0.8, 0.5, 0.5, 0.2, 5, 0.5);
    expectValidResult(result, 2);
  });
});

describe("Greedy", () => {
  it("should initialize without throwing", () => {
    expect(() => new ANMetaL.GreedyMH_Real(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess)).not.toThrow();
  });

  it("should return a valid result from run()", () => {
    const greedy = new ANMetaL.GreedyMH_Real(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess);
    const result = greedy.run(10, 15, 42);
    expectValidResult(result, 2);
  });
});

describe("GreedyWL", () => {
  it("should initialize without throwing", () => {
    expect(() => new ANMetaL.GreedyMH_Real_WithLeap(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess)).not.toThrow();
  });

  it("should return a valid result from run()", () => {
    const greedy = new ANMetaL.GreedyMH_Real_WithLeap(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess);
    const result = greedy.run(10, 15, 42, 0.2, 5, 0.5);
    expectValidResult(result, 2);
  });
});

describe("AFSA", () => {
  it("should initialize without throwing", () => {
    expect(() => new ANMetaL.AFSAMH_Real(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess)).not.toThrow();
  });

  it("should return a valid result from run()", () => {
    const afsa = new ANMetaL.AFSAMH_Real(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess);
    const result = afsa.run(10, 15, false, 0.2, 5, 0.5, 0.3, 1, 0.9, 0.1, 42);
    expectValidResult(result, 2);
  });
});

describe("ABC", () => {
  it("should initialize without throwing", () => {
    expect(() => new ANMetaL.ArtificialBeeColony(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess)).not.toThrow();
  });

  it("should return a valid result from run()", () => {
    const abc = new ANMetaL.ArtificialBeeColony(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess);
    const result = abc.run(10, 20, 5, 42);
    expectValidResult(result, 2);
  });
});

describe("ACO", () => {
  it("should initialize without throwing", () => {
    expect(() => new ANMetaL.AntColony(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess)).not.toThrow();
  });

  it("should return a valid result from run()", () => {
    const aco = new ANMetaL.AntColony(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess);
    const result = aco.run(10, 15, 0.1, 1.0, 2.0, 42);
    expectValidResult(result, 2);
  });
});

describe("Bat", () => {
  it("should initialize without throwing", () => {
    expect(() => new ANMetaL.BatAlgorithm(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess)).not.toThrow();
  });

  it("should return a valid result from run()", () => {
    const bat = new ANMetaL.BatAlgorithm(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess);
    const result = bat.run(10, 15, 0, 2, 0.5, 0.5, 42);
    expectValidResult(result, 2);
  });
});

describe("BlackHole", () => {
  it("should initialize without throwing", () => {
    expect(() => new ANMetaL.BlackHole(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess)).not.toThrow();
  });

  it("should return a valid result from run()", () => {
    const bh = new ANMetaL.BlackHole(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess);
    const result = bh.run(10, 15, 42);
    expectValidResult(result, 2);
  });
});

describe("Cuckoo", () => {
  it("should initialize without throwing", () => {
    expect(() => new ANMetaL.CuckooSearch(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess)).not.toThrow();
  });

  it("should return a valid result from run()", () => {
    const cuckoo = new ANMetaL.CuckooSearch(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess);
    const result = cuckoo.run(10, 15, 0.25, 42);
    expectValidResult(result, 2);
  });
});

describe("Firefly", () => {
  it("should initialize without throwing", () => {
    expect(() => new ANMetaL.FireflyAlgorithm(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess)).not.toThrow();
  });

  it("should return a valid result from run()", () => {
    const firefly = new ANMetaL.FireflyAlgorithm(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess);
    const result = firefly.run(10, 15, 0.5, 1.0, 1.0, 42);
    expectValidResult(result, 2);
  });
});

describe("Harmony", () => {
  it("should initialize without throwing", () => {
    expect(() => new ANMetaL.HarmonySearch(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess)).not.toThrow();
  });

  it("should return a valid result from run()", () => {
    const harmony = new ANMetaL.HarmonySearch(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess);
    const result = harmony.run(10, 15, 0.9, 0.3, 0.2, 42);
    expectValidResult(result, 2);
  });
});

describe("Genetic", () => {
  const categorics: (string | number)[][] = Array.from({ length: 5 }, () => ["is", "not"]);
  const geneticObjFunc = (p: number[]) => (p as unknown as string[]).filter((x) => x === "is").length;
  const geneticRepair = (p: number[]) => p;
  const geneticPreprocess = (p: number[]) => p;

  it("should initialize without throwing", () => {
    expect(() => new ANMetaL.GeneticMH_Categorical(categorics, 5, true, geneticObjFunc, geneticRepair, geneticPreprocess)).not.toThrow();
  });

  it("should return a valid result from run()", () => {
    const genetic = new ANMetaL.GeneticMH_Categorical(categorics, 5, true, geneticObjFunc, geneticRepair, geneticPreprocess);
    const result = genetic.run(10, 15, 0.3, 0.1, true, true, 42);
    expect(result).toHaveProperty("fitness");
    expect(result).toHaveProperty("point");
    expect(typeof result.fitness).toBe("number");
    expect(Number.isFinite(result.fitness)).toBe(true);
    expect(result.point.length).toBe(5);
  });
});

describe("GeneticWithLeap", () => {
  const categorics: (string | number)[][] = Array.from({ length: 5 }, () => ["is", "not"]);
  const geneticObjFunc = (p: number[]) => (p as unknown as string[]).filter((x) => x === "is").length;
  const geneticRepair = (p: number[]) => p;
  const geneticPreprocess = (p: number[]) => p;

  it("should initialize without throwing", () => {
    expect(() => new ANMetaL.GeneticMH_Categorical_WithLeap(categorics, 5, true, geneticObjFunc, geneticRepair, geneticPreprocess)).not.toThrow();
  });

  it("should return a valid result from run()", () => {
    const genetic = new ANMetaL.GeneticMH_Categorical_WithLeap(categorics, 5, true, geneticObjFunc, geneticRepair, geneticPreprocess);
    const result = genetic.run(10, 15, 0.3, 0.1, true, true, 42, false, 0.2, 5, 0.5);
    expect(result).toHaveProperty("fitness");
    expect(result).toHaveProperty("point");
    expect(typeof result.fitness).toBe("number");
    expect(Number.isFinite(result.fitness)).toBe(true);
    expect(result.point.length).toBe(5);
  });
});

describe("runYielded", () => {
  it("PSO with iterations=5 should yield 7 times", () => {
    const pso = new ANMetaL.PSOMH_Real(limits.min[0], limits.max[0], 2, false, objFunc, repair, preprocess);
    const gen = pso.runYielded(5, 15, 42, 0.8, 0.5, 0.5);
    const results: ANMetaL.YieldResult[] = [];
    for (const state of gen) {
      results.push(state);
    }
    expect(results.length).toBe(7);
  });
});
