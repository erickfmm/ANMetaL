export class Sudoku {
  state: number[][];
  categories: number[];

  constructor(initialState?: number[][]) {
    if (initialState !== undefined) {
      this.state = initialState;
    } else {
      this.state = Array.from({ length: 9 }, () =>
        new Array<number>(9).fill(0)
      );
    }
    this.categories = [1, 2, 3, 4, 5, 6, 7, 8, 9];
  }

  setCategories(categories: number[]): void {
    this.categories = categories;
  }

  isRowViolating(row: number[]): boolean {
    if (row.length !== 9) {
      throw new Error("Row should be of size 9");
    }
    for (let i = 0; i < row.length; i++) {
      const x = row[i];
      if (x !== 0 && !this.categories.includes(x)) return true;
      if (x !== 0) {
        let count = 0;
        for (const y of row) {
          if (y === x) count++;
        }
        if (count > 1) return true;
      }
    }
    return false;
  }

  getViolations(): number {
    let nViolations = 0;
    for (let r = 0; r < 9; r++) {
      if (this.isRowViolating(this.state[r])) nViolations++;
    }
    for (let c = 0; c < 9; c++) {
      const col = this.state.map((row) => row[c]);
      if (this.isRowViolating(col)) nViolations++;
    }
    for (const br of [0, 3, 6]) {
      for (const bc of [0, 3, 6]) {
        const box: number[] = [];
        for (let r = br; r < br + 3; r++) {
          for (let c = bc; c < bc + 3; c++) {
            box.push(this.state[r][c]);
          }
        }
        if (this.isRowViolating(box)) nViolations++;
      }
    }
    return nViolations;
  }

  isValidSolution(solution?: number[][]): boolean {
    if (solution !== undefined) {
      const oldState = this.state;
      this.state = solution;
      const valid = this.getViolations() === 0;
      this.state = oldState;
      return valid;
    }
    return this.getViolations() === 0;
  }
}
