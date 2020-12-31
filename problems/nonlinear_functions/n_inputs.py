#functions taken from https://doi.org/10.1007/s00521-017-3088-3
#Applying modified cuckoo search algorithm for solving systems of nonlinear equations
import numpy as np

#all were tested in the paper with 20 dimensions
class Brown1:
    @staticmethod
    def func(x_array: list) -> float:
        result = 0.0
        for x in x_array:
            result += x-3
        result = result**2
        for i in range(0, len(x_array)-1):
            result += pow(10, -3)*((x_array[i]-3)**2) - (x_array[i]-x_array[i+1])+np.exp(20*(x_array[i]-x_array[i+1]))
        return result

    @staticmethod
    def get_limits():
        return [-1, 4]
    
    @staticmethod
    def get_theoretical_optimum():
        return 2.0
    
    @staticmethod
    def get_type():
        return "min"

class Brown3:
    @staticmethod
    def func(x_array: list) -> float:
        result = 0.0
        for i in range(0, len(x_array)-1):
            result += pow((x_array[i])**2, (x_array[i+1]**2)+1)
            result += pow((x_array[i+1])**2, (x_array[i]**2)+1)
        return result

    @staticmethod
    def get_limits():
        return [-1, 4]
    
    @staticmethod
    def get_theoretical_optimum():
        return 0.0
    
    @staticmethod
    def get_type():
        return "min"

class F10n:
    @staticmethod
    def func(x_array: list) -> float:
        result = 0.0
        for i in range(0, len(x_array)-1):
            result += (x_array[i]-1)**2 * (1 + 10*(np.sin(np.pi*x_array[i+1]))**2)
        result += 10*(np.sin(np.pi * x_array[0])**2)
        result += (x_array[len(x_array)-1] - 1)**2
        result = (np.pi/ len(x_array)) * result
        return result

    @staticmethod
    def get_limits():
        return [-10, 10]
    
    @staticmethod
    def get_theoretical_optimum():
        return 0.0
    
    @staticmethod
    def get_type():
        return "min"

class F15n:
    @staticmethod
    def func(x_array: list) -> float:
        result = 0.0
        for i in range(0, len(x_array)-1):
            result += (x_array[i]-1)**2 * (1 + (np.sin(3*np.pi*x_array[i+1]))**2)
        result += (np.sin(3*np.pi * x_array[0])**2)
        result += (1/10.0) * (x_array[len(x_array)-1] - 1)**2 * (1+ np.sin(2*np.pi*x_array[len(x_array)-1]))
        result = (1/10.0) * result
        return result

    @staticmethod
    def get_limits():
        return [-10, 10]
    
    @staticmethod
    def get_theoretical_optimum():
        return 0.0
    
    @staticmethod
    def get_type():
        return "min"



#functions taken from https://doi.org/10.1007/s00521-018-3512-3
#An Improved dynamic self-adaption cuckoo search algorithm based on collaboration between subpopulations

class Sphere:
    @staticmethod
    def func(x_array: list) -> float:
        result = 0.0
        for x in x_array:
            result += x**2
        return result

    @staticmethod
    def get_limits():
        return [-100, 100]
    
    @staticmethod
    def get_theoretical_optimum():
        return 0.0
    
    @staticmethod
    def get_type():
        return "min"

class Rosenbrock:
    @staticmethod
    def func(x_array: list) -> float:
        result = 0.0
        for i in range(0, len(x_array)-1):
            result += (100.0*(x_array[i+1] - (x_array[i]**2))**2) + ((x_array[i] - 1)**2)
        return result

    @staticmethod
    def get_limits():
        return [-30, 30]
    
    @staticmethod
    def get_theoretical_optimum():
        return 0.0
    
    @staticmethod
    def get_type():
        return "min"


class Griewank:
    @staticmethod
    def func(x_array: list) -> float:
        r_sum = 0.0
        for x in x_array:
            r_sum += x**2
        r_prod = 1.0
        for i in range(len(x_array)):
            r_prod *= np.cos(x_array[i]/float(np.sqrt(i+1)))
        result = (1/4000.0)*r_sum - r_prod + 1.0
        return result

    @staticmethod
    def get_limits():
        return [-600, 600]
    
    @staticmethod
    def get_theoretical_optimum():
        return 0.0
    
    @staticmethod
    def get_type():
        return "min"


class Rastrigrin:
    @staticmethod
    def func(x_array: list) -> float:
        result = 0.0
        for x in x_array:
            result += x**2 - 10*np.cos(2 * np.pi * x) +10
        return result

    @staticmethod
    def get_limits():
        return [- 5.12, 5.12]
    
    @staticmethod
    def get_theoretical_optimum():
        return 0.0
    
    @staticmethod
    def get_type():
        return "min"


class Sumsquares:
    @staticmethod
    def func(x_array: list) -> float:
        result = 0.0
        for i in range(len(x_array)):
            result += (i+1)*(x_array[i]**2)
        return result

    @staticmethod
    def get_limits():
        return [-10, 10] 
    
    @staticmethod
    def get_theoretical_optimum():
        return 0.0
    
    @staticmethod
    def get_type():
        return "min"


class Michalewicz:
    @staticmethod
    def func(x_array: list, m:float = 10) -> float:
        result = 0.0
        for i in range(len(x_array)):
            result += np.sin(x_array[i]) * ((np.sin( ((i+1)*(x_array[i]**2))/ np.pi ))**(2*m))
        result = -result
        return result

    @staticmethod
    def get_limits():
        return [0, np.pi]
    
    @staticmethod
    def get_theoretical_optimum():
        #-1.8013 if d=2
        #-4.687658 if d=5
        #-9.66015 if d=10
        return 0.0
    
    @staticmethod
    def get_type():
        return "min"


class Quartic:
    @staticmethod
    def func(x_array: list) -> float:
        result = 0.0
        for i in range(len(x_array)):
            result += (i+1)*(x_array[i]**4) + np.random.random()
        return result

    @staticmethod
    def get_limits():
        return [- 1.28, 1.28]
    
    @staticmethod
    def get_theoretical_optimum():
        return 0.0
    
    @staticmethod
    def get_type():
        return "min"


class Schwefel:
    @staticmethod
    def func(x_array: list) -> float:
        result = 0.0
        for x in x_array:
            result += x * np.sin(np.sqrt(np.abs(x)))
        result = ( 418.9829 * len(x_array) ) - result
        return result

    @staticmethod
    def get_limits():
        return [-500, 500]
    
    @staticmethod
    def get_theoretical_optimum():
        return 0.0#- 418.9 * 5.0 #- 418.9 X 5
    
    @staticmethod
    def get_type():
        return "min"


class Penalty:
    @staticmethod
    def func(x_array: list) -> float:
        def y_i(x_i):
            return 1.0 + ( (x_i+1.0)/4.0 )
        def u(x_i, a, k, m):
            if x_i > a:
                return k * ((x_i - a)**m)
            elif -a <= x_i <= a:
                return 0
            else: # x_i < -a
                return k * ((-x_i - a)**m)
        sum_u = 0.0
        for x in x_array:
            sum_u += u(x, 10, 100, 4)
        result = 0.0
        for i in range(len(x_array)-1):
            result += (y_i(x_array[i]) - 1)**2 * ( 1 + 10*np.sin(np.pi * y_i(x_array[i+1]))**2 )
        result += 10 * np.sin(np.pi*y_i(x_array[0]))
        result += (y_i(x_array[len(x_array)-1]) + 1)**2
        result = (np.pi / float(len(x_array))) * result
        result += sum_u
        return result

    @staticmethod
    def get_limits():
        return [-50, 50]
    
    @staticmethod
    def get_theoretical_optimum():
        return 0.0
    
    @staticmethod
    def get_type():
        return "min"
