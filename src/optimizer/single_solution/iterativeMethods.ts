export function eulerMethod(
  dyTimesDxFunc: (x: number, y: number) => number,
  xStart: number,
  xEnd: number,
  y0: number,
  nSteps: number
): number {
  const h = (xEnd - xStart) / nSteps;
  let x = xStart;
  let y = y0;
  for (let i = 0; i < nSteps; i++) {
    y = y + h * dyTimesDxFunc(x, y);
    x = x + h;
  }
  return y;
}

export function newtonMethod(
  func: (x: number) => number,
  derivativeFunc: (x: number) => number,
  xStart: number,
  nIterations: number,
  m: number = 1
): number {
  let x = xStart;
  for (let i = 0; i < nIterations; i++) {
    x = x - m * (func(x) / derivativeFunc(x));
  }
  return x;
}
