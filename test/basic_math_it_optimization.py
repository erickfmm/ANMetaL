import math
import anmetal.iterative_methods as it_optimization


##example of newton method for 100^100 in video of "derivando" channel
class x_times_x:
    def __init__(self, result=0):
        self.result = result
    def __call__(self, x):
        return pow(x, x) - self.result

#def x_times_x(x, result=0):
#    return pow(x, x) - result

def x_times_x_derivative(x):
    return pow(x, x) * (1 + math.log(x))

x_times_x100_func = x_times_x(100)
x_times_x100_res = it_optimization.newton_method(x_times_x100_func, x_times_x_derivative, 3.5, 20)
print("x^x = 100, then x is aprox:",x_times_x100_res, " what gives: ", pow(x_times_x100_res, x_times_x100_res))


##example of euler method in wikipedia

def dy_dx(x, y):
    return math.sin(x) - math.log(y)

#exact value es y=0.3325459
y_res = it_optimization.euler_method(dy_dx, 0.13, 0.14, 0.32, 4)
print("euler example: ", y_res)

def dy_dx_line(x, y):
    #y = mx+b
    #dy = m
    return 2

y_res = it_optimization.euler_method(dy_dx_line, 6, 8, 25, 4) #y = x*2+13, 25=6*2+13
print("euler example 2 (29=8*2+13): ", y_res) # 29