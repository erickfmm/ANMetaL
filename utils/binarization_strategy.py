
def standard(x: float, uniform_value: float):
    return 1 if uniform_value <= x else 0

def complement(x: float, uniform_value1: float, uniform_value2: float):
    return standard(1 - x, uniform_value2) if uniform_value1 <= x else 0

def static_probability(x: float, alpha: float, uniform_value: float):
    if alpha >= x:
        return 0
    else:
        if (alpha < x and x <= ((1 + alpha) / 2)):
            return standard(x, uniform_value)
        else:
            return 1

def elitist(x: float, uniform_value1: float, uniform_value2: float):
    return standard(x, uniform_value2) if uniform_value1 < x else 0
