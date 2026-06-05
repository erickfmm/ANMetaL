export function nsphereToCartesian(r: number, angles: number[]): number[] {
  const n = angles.length + 1;
  const a = new Array(n);
  a[0] = 2 * Math.PI;
  for (let i = 1; i < n; i++) {
    a[i] = angles[i - 1];
  }
  const si = new Array(n);
  si[0] = 1;
  for (let i = 0; i < n; i++) {
    si[i] = i === 0 ? 1 : si[i - 1] * Math.sin(a[i]);
  }
  const co = new Array(n);
  for (let i = 0; i < n; i++) {
    co[i] = Math.cos(a[i]);
  }
  const coRolled = new Array(n);
  for (let i = 0; i < n - 1; i++) {
    coRolled[i] = co[i + 1];
  }
  coRolled[n - 1] = co[0];
  const result = new Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = si[i] * coRolled[i] * r;
  }
  return result;
}

export function distance(point1: number[], point2: number[]): number {
  if (point1.length !== point2.length) {
    throw new ValueError();
  }
  let sumPows = 0;
  for (let i = 0; i < point1.length; i++) {
    sumPows += Math.pow(point2[i] - point1[i], 2);
  }
  return Math.sqrt(sumPows);
}

export function distanceSquared(point1: number[], point2: number[]): number {
  if (point1.length !== point2.length) {
    throw new ValueError();
  }
  let sumPows = 0;
  for (let i = 0; i < point1.length; i++) {
    sumPows += Math.pow(point2[i] - point1[i], 2);
  }
  return sumPows;
}

export function distanceTaxicab(point1: number[], point2: number[]): number {
  if (point1.length !== point2.length) {
    throw new ValueError();
  }
  let sumAbs = 0;
  for (let i = 0; i < point1.length; i++) {
    sumAbs += Math.abs(point2[i] - point1[i]);
  }
  return sumAbs;
}

class ValueError extends Error {
  constructor() {
    super("points must have the same length");
    this.name = "ValueError";
  }
}
