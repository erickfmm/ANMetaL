from anmetal.problems.IProblem import IProblem

import numpy as np
from numpy.random import RandomState
class Partition_Subset_Abstract(IProblem):
    def __init__(self, seed:int, num_dims:int, data=None):
        if data is not None and type(data) == type([]):
            self.data = data
        else:
            rnd = RandomState(seed)
            self.data = rnd.uniform(0, num_dims-1, num_dims)
            #mydata = rnd.randint(-100, 100, 200)
        self.ndim = len(self.data)+2
        self.min_x = 0
        self.max_x = len(self.data)


    def is_valid(self, point):
        are_1s = [0 for _ in range(len(self.data))]
        for i in range(2, len(point)):
            if point[i] < 0 or int(point[i]) >= len(self.data):
                return False
            are_1s[int(point[i])] += 1
            if are_1s[int(point[i])] > 1:
                return False
        for el in are_1s:
            if el == 0 or el > 1:
                return False
        return True

    def repair_function(self, point):
        point = np.abs(point)
        are_not = []
        are_1s = [0 for _ in range(len(self.data))]
        #dups = []
        dups = [[] for _ in range(len(self.data))]
        for i in range(2, len(point)):
            if int(point[i]) < 0 or int(point[i]) >= len(self.data):
                point[i] = 0
            are_1s[int(point[i])] += 1
            dups[int(point[i])].append(i)
        for i in range(len(are_1s)):
            if are_1s[i] == 0:
                are_not.append(i)
        sum_dups = []
        for i in range(len(dups)):
            sum_dups.append(len(dups[i]))
            if len(dups[i]) > 0:
                del dups[i][0]
        dups = [e for di in dups for e in di] #flatten
        if len(are_not) != len(dups) or are_1s != sum_dups:
            print("rayooooooooooos son distintos")
            print("arenot: ", len(are_not), ", dups: ", len(dups))
            print("are not: ", are_not)
            print("dups: ", dups)
            print("are1s: ", are_1s)
            point2 = [int(e) for e in point]
            print("point int: ", point2)
            print("point: ", point)
            raise Exception("son distintos :c")
        for i in range(len(dups)):
            point[dups[i]] = are_not[i]
        return point

    def preprocess_function(self, point):
        point2 = np.abs(np.copy(point))
        for i in range(2, len(point2)):
            point2[i] = int(point2[i])
        return point2

    def objective_function(self, point):
        raise NotImplementedError
    
    def get_sums(self, point):
        if not self.is_valid(point):
            return (False, False)
        s1 = point[0]
        s2 = point[1]
        s1 = int(s1/float(s1+s2)*len(self.data))
        s1 = s1 if s1 >= 0 and s1 < len(self.data) else 0
        s2 = len(self.data)-s1
        sum1 = 0
        sum2 = 0
        for i in range(2, s1+2, 1):
            sum1 += self.data[int(point[i])]
        for i in range(s1+2, len(point), 1):
            sum2 += self.data[int(point[i])]
        return (sum1, sum2)

    def print_difference_subsets(self, point):
        s1 = point[0]
        s2 = point[1]
        s1 = int(s1/float(s1+s2)*len(self.data))
        s1 = s1 if s1 >= 0 and s1 < len(self.data) else 0
        s2 = len(self.data)-s1
        sum1 = 0
        sum2 = 0
        sub1 = []
        sub2 = []
        for i in range(2, s1+2, 1):
            sum1 += self.data[int(point[i])]
            sub1.append(self.data[int(point[i])])
        for i in range(s1+2, len(point), 1):
            sum2 += self.data[int(point[i])]
            sub2.append(self.data[int(point[i])])
        print("sum1:", sum1)
        print("sum2:", sum2)

class Partition_Real(Partition_Subset_Abstract):
    #def __init__(self, seed:int, num_dims:int, data=None):
    #    super().__init__(seed, num_dims, data)

    def objective_function(self, point):
        sum1, sum2 = self.get_sums(point)
        if sum1 is False:
            return False
        return abs(sum1-sum2)


class Subset_Real(Partition_Subset_Abstract):
    #def __init__(self, seed:int, num_dims:int, data=None):
    #    super().__init__(seed, num_dims, data)

    def objective_function(self, point):
        sum1, _ = self.get_sums(point)
        if sum1 is False:
            return False
        return np.abs(sum1)
