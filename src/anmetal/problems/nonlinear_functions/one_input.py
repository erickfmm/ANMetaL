#functions taken from https://doi.org/10.1007/s00521-017-3088-3
#Applying modified cuckoo search algorithm for solving systems of nonlinear equations
import numpy as np

class F1:
    @staticmethod
    def func_simple(x: float) -> float:
        result = 2 * (x - 0.75)**2
        result = result + np.sin(5*np.pi*x - 0.4*np.pi)
        result = result - 0.125
        return result
    
    @staticmethod
    def func(x_array: list) -> float:
        return F1.func_simple(x_array[0])

    @staticmethod
    def get_limits():
        return [0, 1]
    
    @staticmethod
    def get_theoretical_optimum():
        return -1.12323
    
    @staticmethod
    def get_type():
        return "min"

class F3:
    @staticmethod
    def func_simple(x: float) -> float:
        result = 0
        for j in range(1, 5+1): #j = 1, 2, 3, 4, 5
            result += j * np.sin( (j+1)*x +j )
        return -1*result

    @staticmethod
    def func(x_array: list) -> float:
        return F3.func_simple(x_array[0])

    @staticmethod
    def get_limits():
        return [-10, 10]
    
    @staticmethod
    def get_theoretical_optimum():
        return -12.03125
    
    @staticmethod
    def get_type():
        return "min"
