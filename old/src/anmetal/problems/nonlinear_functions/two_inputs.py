#functions taken from https://doi.org/10.1007/s00521-017-3088-3
#Applying modified cuckoo search algorithm for solving systems of nonlinear equations
import numpy as np

class Camelback:
    @staticmethod
    def func_simple(x: float, y:float) -> float:
        result = (4 - 2.1*(x**2) + (x**4)/3.0) * (x**2)
        result += x*y
        result += (4*(y**2) - 4) * (y**2)
        return result

    @staticmethod
    def func(x_array: list) -> float:
        return Camelback.func_simple(x_array[0], x_array[1])

    @staticmethod
    def get_limits():
        return [-3, 3], [-2, 2]
    
    @staticmethod
    def get_theoretical_optimum():
        return -1.03163
    
    @staticmethod
    def get_type():
        return "min"
    
class Goldsteinprice:
    @staticmethod
    def func_simple(x: float, y:float) -> float:
        result = 1 + ((x+y+1)**2) * (19-14*x + 3*(x**2)-14*y + 6*x*y + 3*(y**2))
        result = result * (30 + ((2*x - 3*y)**2) * (18 - 32*x + 12*(x**2) + 48*y - 36*x*y + 27*(y**2)))
        return result

    @staticmethod
    def func(x_array: list) -> float:
        return Goldsteinprice.func_simple(x_array[0], x_array[1])
        
    @staticmethod
    def get_limits():
        return [-2, 2]
    
    @staticmethod
    def get_theoretical_optimum():
        return 3
    
    @staticmethod
    def get_type():
        return "min"

class Pshubert1:
    @staticmethod
    def func_simple(x: float, y:float) -> float:
        res1 = 0
        for i in range(1, 6):#1, 2, 3, 4, 5
            res1 += float(i)*np.cos( (i+1.0)*x + i )
        res2 = 0
        for i in range(1, 6):#1,2,3,4,5
            res2 += float(i)*np.cos( (i+1.0)*y + i )
        result = (0.5*( (x+1.42513)**2 + (y+0.80032)**2 ) )
        result = (res1*res2) + result
        return result

    @staticmethod
    def func(x_array: list) -> float:
        return Pshubert1.func_simple(x_array[0], x_array[1])
        
    @staticmethod
    def get_limits():
        return [-10, 10]
    
    @staticmethod
    def get_theoretical_optimum():
        return -186.73091
    
    @staticmethod
    def get_type():
        return "min"

class Pshubert2:
    @staticmethod
    def func_simple(x: float, y:float) -> float:
        res1 = 0
        for i in range(1, 6):#1, 2, 3, 4, 5
            res1 += float(i)*np.cos( (i+1.0)*x + i )
        res2 = 0
        for i in range(1, 6):#1,2,3,4,5
            res2 += float(i)*np.cos( (i+1.0)*y + i )
        result = (x+1.42513)**2 + (y+0.80032)**2
        result = (res1*res2) + result
        return result

    @staticmethod
    def func(x_array: list) -> float:
        return Pshubert2.func_simple(x_array[0], x_array[1])
        
    @staticmethod
    def get_limits():
        return [-10, 10]
    
    @staticmethod
    def get_theoretical_optimum():
        return -186.73091
    
    @staticmethod
    def get_type():
        return "min"
        
class Shubert:
    @staticmethod
    def func_simple(x: float, y:float) -> float:
        res1 = 0
        for i in range(1, 6):#1, 2, 3, 4, 5
            res1 += float(i)*np.cos( (i+1.0)*x + i )
        res2 = 0
        for i in range(1, 6):#1,2,3,4,5
            res2 += float(i)*np.cos( (i+1.0)*y + i )
        return res1*res2

    @staticmethod
    def func(x_array: list) -> float:
        return Shubert.func_simple(x_array[0], x_array[1])
        
    @staticmethod
    def get_limits():
        return [-10, 10]
    
    @staticmethod
    def get_theoretical_optimum():
        return -186.73091
    
    @staticmethod
    def get_type():
        return "min"
        
class Quartic:
    @staticmethod
    def func_simple(x: float, y:float) -> float:
        return (x**4/float(4)) - (x**2/float(2)) + (x/float(10)) + (y**2/float(2))

    @staticmethod
    def func(x_array: list) -> float:
        return Quartic.func_simple(x_array[0], x_array[1])
        
    @staticmethod
    def get_limits():
        return [-10, 10]
    
    @staticmethod
    def get_theoretical_optimum():
        return -0.35239
    
    @staticmethod
    def get_type():
        return "min"