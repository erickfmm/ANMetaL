

def euler_method(dy_times_dx_func, x_start, x_end, y0, n_steps):
    h = float(x_end - x_start) / float(n_steps)
    x = x_start
    y = y0
    for _ in range(n_steps):
        y = y + h * dy_times_dx_func(x, y)
        x = x + h
    return y

def newton_method(func, derivative_func, x_start, n_iterations, m=1):
    x = x_start
    for _ in range(n_iterations):
        x = x - m*(float(func(x)) / float(derivative_func(x)))
    return x