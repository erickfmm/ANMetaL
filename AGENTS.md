# ANMetaL — Agent Notes

## Commands

- `npm run build` — Rollup bundles `src/index.ts` → `dist/anmetal.min.js` (UMD, global `ANMetaL`) + `dist/anmetal.esm.js` (ESM), both minified. Also emits `dist/*.d.ts` declarations.
- `npm test` — **Builds first**, then runs vitest (`npm run build && vitest run`). Tests import from `"anmetal"` which is aliased to the compiled ESM bundle — source edits are NOT tested directly.
- `npm run test:watch` — Runs vitest in watch mode. Still requires a prior `npm run build` for the alias to resolve.
- No lint, format, typecheck, or CI commands exist.

## Architecture

- **Zero runtime deps.** All math is native JS/TS. Dev deps only: rollup, typescript, vitest, tslib, terser.
- **Entry point:** `src/index.ts` — barrel-exports 30+ symbols (algorithms, problems, utilities, types).
- **Algorithm pattern:** Every optimizer extends `IMetaheuristic` (`src/optimizer/IMetaheuristic.ts`). Constructor takes `(minValue, maxValue, ndims, toMax, objectiveFunction, repairFunction, preprocessFunction)`. Exposes `run()` → `RunResult` and `runYielded()` → `Generator<YieldResult>` for iteration tracking.
- **Problem pattern:** Benchmark functions are static classes (`static func()`, `static getLimits()`, `static getTheoreticalOptimum()`, `static getType()`). NP-hard problems extend `IProblem` with instance methods (`objectiveFunction`, `repairFunction`, `preprocessFunction`).
- **PRNG:** `SeededRandom` uses mulberry32 algorithm. All algorithms accept a `seed` param for reproducibility.

## Key Directories

- `src/optimizer/population/` — 12 population-based algorithms, each in its own subdirectory (pso, greedy, afsa, abc, aco, bat, blackhole, cuckoo, firefly, harmony, genetic)
- `src/optimizer/single_solution/` — `eulerMethod`, `newtonMethod`
- `src/problems/nonlinear_functions/` — Split into `oneInput.ts`, `twoInputs.ts`, `nInputs.ts`
- `src/problems/nphard_real/` — `PartitionReal`, `SubsetReal` (share `PartitionSubsetAbstract`)
- `src/problems/nphard_categorical/` — `KnapsackCategorical`, `Sudoku`, `SudokuOptimized`
- `tests/` — 4 vitest files testing compiled output (~533 lines total, ~81 tests)

## Build + Test Gotchas

- **Tests run against compiled output, not source.** The vitest config aliases `"anmetal"` → `dist/anmetal.esm.js`. You MUST build before testing. `npm test` handles this, but `vitest run` alone will fail if `dist/` is stale or missing.
- `dist/` is gitignored — a fresh clone needs `npm run build` before `npm test` works.
- TypeScript strict mode is enabled. No source maps in build output.

## Browser Visualizer

- `index.html` (1601 lines) — self-contained ANMetaL Visualizer app at repo root. Loads `anmetal.min.js` via `<script>` and accesses `window.ANMetaL`. External CDN deps: Tailwind CSS, Chart.js v4, Three.js r128, JSZip, FileSaver.
- Three tabs: "MH Graph Each It" (2D/3D population viz on benchmark functions with heatmap overlay), "Trajectory Plot NPComp" (multi-algorithm convergence comparison on NP-hard problems), "Genetic Categorical Plot" (genetic algorithm on categorical problems with population grid rendering).
- All tabs support play/pause animation via `runYielded()` iteration history, export to ZIP/WebM/MP4.
- To use: `npm run build`, then open `index.html` in a browser (requires `dist/anmetal.min.js` to exist).

## Conventions

- No ESLint, Prettier, or other code style tooling configured.
- No CI/CD pipeline.
- Legacy Python version referenced in README (`old/` directory) does not exist in the current tree (excluded by tsconfig and likely removed).
- `tsconfig.json` excludes `tests/`, `old/`, `dist/` from compilation.
