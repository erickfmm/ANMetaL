import math

def sShape1(x: float):
    return (1 / (1 + math.pow(math.e, -2 * x)))

def sShape2(x: float):
    return (1 / (1 + math.pow(math.e, -x)))

def sShape3(x: float):
    return (1 / (1 + math.pow(math.e, -x / 2)))

def sShape4(x: float):
    return (1 / (1 + math.pow(math.e, -x / 3)))

def vShape1(x: float):
    return abs(erf((math.sqrt(math.pi) / 2) * x))

def vShape2(x: float):
    return abs(math.tanh(x))

def vShape3(x: float):
    return abs(x / math.sqrt(1 + math.pow(x, 2)))

def vShape4(x: float):
    return abs((2 / math.pi) * math.atan((math.pi / 2) * x))

def erf(z: float):
    q: float = 1.0 / (1.0 + 0.5 * abs(z))
    ans: float = 1 - q * math.exp(-z * z - 1.26551223
            + q * (1.00002368
            + q * (0.37409196
            + q * (0.09678418
            + q * (-0.18628806
            + q * (0.27886807
            + q * (-1.13520398
            + q * (1.48851587
            + q * (-0.82215223
            + q * (0.17087277))))))))))
    return ans if z >= 0 else -ans
