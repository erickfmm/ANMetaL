export function standard(x: number, uniformValue: number): 0 | 1 {
  return uniformValue <= x ? 1 : 0;
}

export function complement(
  x: number,
  uniformValue1: number,
  uniformValue2: number,
): 0 | 1 {
  if (uniformValue1 <= x) {
    return standard(1 - x, uniformValue2);
  }
  return 0;
}

export function staticProbability(
  x: number,
  alpha: number,
  uniformValue: number,
): 0 | 1 {
  if (alpha >= x) {
    return 0;
  } else {
    if (alpha < x && x <= (1 + alpha) / 2) {
      return standard(x, uniformValue);
    } else {
      return 1;
    }
  }
}

export function elitist(
  x: number,
  uniformValue1: number,
  uniformValue2: number,
): 0 | 1 {
  if (uniformValue1 < x) {
    return standard(x, uniformValue2);
  }
  return 0;
}
