export function sShape1(x: number): number {
  return 1 / (1 + Math.pow(Math.E, -2 * x));
}

export function sShape2(x: number): number {
  return 1 / (1 + Math.pow(Math.E, -x));
}

export function sShape3(x: number): number {
  return 1 / (1 + Math.pow(Math.E, -x / 2));
}

export function sShape4(x: number): number {
  return 1 / (1 + Math.pow(Math.E, -x / 3));
}

export function vShape1(x: number): number {
  return Math.abs(erf((Math.sqrt(Math.PI) / 2) * x));
}

export function vShape2(x: number): number {
  return Math.abs(Math.tanh(x));
}

export function vShape3(x: number): number {
  return Math.abs(x / Math.sqrt(1 + Math.pow(x, 2)));
}

export function vShape4(x: number): number {
  return Math.abs((2 / Math.PI) * Math.atan((Math.PI / 2) * x));
}

export function erf(z: number): number {
  const q: number = 1.0 / (1.0 + 0.5 * Math.abs(z));
  const ans: number =
    1 -
    q *
      Math.exp(
        -z * z -
          1.26551223 +
          q *
            (1.00002368 +
              q *
                (0.37409196 +
                  q *
                    (0.09678418 +
                      q *
                        (-0.18628806 +
                          q *
                            (0.27886807 +
                              q *
                                (-1.13520398 +
                                  q *
                                    (1.48851587 +
                                      q * (-0.82215223 + q * 0.17087277)))))))),
      );
  return z >= 0 ? ans : -ans;
}
