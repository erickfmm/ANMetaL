import { IProblem } from "../IProblem";

export class SudokuOptimized extends IProblem {
  originalState: number[][];
  state: number[][];
  fixedMask: boolean[][];
  categories: number[][];

  constructor(initialState?: number[][]) {
    super();
    if (initialState !== undefined) {
      this.originalState = initialState.map((row) => [...row]);
    } else {
      this.originalState = Array.from({ length: 9 }, () =>
        new Array<number>(9).fill(0)
      );
    }
    this.state = this.originalState.map((row) => [...row]);
    this.preprocessConstraints();
    this.fixedMask = this.state.map((row) => row.map((v) => v !== 0));
    this.categories = [];
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        if (this.fixedMask[r][c]) {
          this.categories.push([this.state[r][c]]);
        } else {
          this.categories.push([1, 2, 3, 4, 5, 6, 7, 8, 9]);
        }
      }
    }
  }

  preprocessConstraints(): void {
    let changed = true;
    while (changed) {
      changed = false;
      for (let r = 0; r < 9; r++) {
        for (let c = 0; c < 9; c++) {
          if (this.state[r][c] === 0) {
            const possibles = this.getPossibleValues(r, c);
            if (possibles.length === 1) {
              this.state[r][c] = possibles[0];
              changed = true;
            }
          }
        }
      }
    }
  }

  getPossibleValues(r: number, c: number): number[] {
    const rowVals = this.state[r];
    const colVals = this.state.map((row) => row[c]);
    const br = Math.floor(r / 3) * 3;
    const bc = Math.floor(c / 3) * 3;
    const boxVals: number[] = [];
    for (let i = br; i < br + 3; i++) {
      for (let j = bc; j < bc + 3; j++) {
        boxVals.push(this.state[i][j]);
      }
    }
    const used = new Set([...rowVals, ...colVals, ...boxVals]);
    used.delete(0);
    return [1, 2, 3, 4, 5, 6, 7, 8, 9].filter((v) => !used.has(v));
  }

  getCategories(): number[][] {
    return this.categories;
  }

  objectiveFunction(point: number[]): number | false {
    const grid: number[][] = [];
    for (let r = 0; r < 9; r++) {
      grid.push(point.slice(r * 9, r * 9 + 9));
    }
    let violations = 0;
    for (let r = 0; r < 9; r++) {
      violations += 9 - new Set(grid[r]).size;
    }
    for (let c = 0; c < 9; c++) {
      const col = grid.map((row) => row[c]);
      violations += 9 - new Set(col).size;
    }
    return violations;
  }

  preprocessFunction(point: number[]): number[] {
    return point;
  }

  repairFunction(point: number[]): number[] {
    const grid = point.slice();
    for (const br of [0, 3, 6]) {
      for (const bc of [0, 3, 6]) {
        const currentVals: number[] = [];
        for (let r = br; r < br + 3; r++) {
          for (let c = bc; c < bc + 3; c++) {
            currentVals.push(grid[r * 9 + c]);
          }
        }
        const presentSet = new Set(currentVals);
        const missingNums = [1, 2, 3, 4, 5, 6, 7, 8, 9].filter(
          (n) => !presentSet.has(n)
        );
        if (missingNums.length === 0) continue;
        const counts: Record<number, number> = {};
        for (const x of currentVals) {
          counts[x] = (counts[x] || 0) + 1;
        }
        const missingIter = missingNums[Symbol.iterator]();
        for (let r = br; r < br + 3; r++) {
          for (let c = bc; c < bc + 3; c++) {
            if (!this.fixedMask[r][c]) {
              const val = grid[r * 9 + c];
              if (counts[val] > 1) {
                const result = missingIter.next();
                if (!result.done) {
                  grid[r * 9 + c] = result.value;
                  counts[val]--;
                }
              }
            }
          }
        }
      }
    }
    return grid;
  }
}
