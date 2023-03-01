from problems.IProblem import IProblem

from numpy.random import RandomState

class Knapsack_Categorical(IProblem):
    def __init__(self, knapsack_capacity, total_posible_elements, seed, max_cost, max_value, elements=None):
        self.knapsack_capacity = knapsack_capacity
        if elements is not None and type(elements) == type([]):
            self.elements = elements
            self.total_posible_elements = len(elements)
        else:
            self.total_posible_elements = total_posible_elements
            rnd = RandomState(seed)
            #list of tuples with value and cost
            self.elements = []
            #fill elements
            for _ in range(total_posible_elements):
                self.elements.append((rnd.uniform(0, max_value), rnd.uniform(0, max_cost)))

    def get_possible_categories(self):
        #categories
        categories = ["is", "not"]
        categories_all_elements = [categories for _ in range(self.total_posible_elements)]
        return categories_all_elements

    def objective_function(self, point):
        if not self.is_valid(point):
            return False
        total_sum = 0
        for i in range(len(point)):
            if point[i] == "is":
                total_sum += self.elements[i][0]
        return total_sum

    def preprocess_function(self, point):
        return point

    def repair_function(self, point):
        i = -1
        times_founded = 0
        while not self.is_valid(point):
            found = False
            while not found:
                i += 1
                if i >= len(point):
                    print("que pedo wey, i es mayor que len en repair")
                    print("i value: ", i)
                    print("times founded: ", times_founded)
                    print("punto: ", point)
                    sum_elements_in, elements_value_in, elements_cost_in = self.get_values_in(point, True)
                    print(sum_elements_in)
                    print(elements_value_in)
                    print(elements_cost_in)
                    exit()
                if point[i] == "is":
                    point[i] = "not"
                    found = True
                    times_founded +=1
        return point    
   
    def get_values_in(self, point, verbose):
        sum_elements_in = 0
        elements_value_in = []
        elements_cost_in = []
        for i in range(len(point)):
            if point[i] == "is":
                sum_elements_in +=1
                elements_value_in.append(self.elements[i][0])
                elements_cost_in.append(self.elements[i][1])
        if verbose:
            print("sum: ", sum_elements_in)
            print("value total: ", sum(elements_value_in))
            print("cost total: ", sum(elements_cost_in))
            print("max knapsack capacity: ", self.knapsack_capacity)
            print("values: ", elements_value_in)
            print("costs: ", elements_cost_in)
        return sum_elements_in, elements_value_in, elements_cost_in

    def is_valid(self, point) -> bool:
        total_sum = 0
        for i in range(len(point)):
            if point[i] == "is":
                total_sum += self.elements[i][1]
        if total_sum > self.knapsack_capacity:
            return False
        return True