class SolutionBasic:
    def __init__(self, point, fitness):
        self.point = point
        self.fitness = fitness

    def move_to(self, new_point, new_fitness):
        self.point = new_point
        self.fitness = new_fitness
        return self.point
    

class SolutionWithId(SolutionBasic):
    def __init__(self, _id, point, fitness):
        self.point = point
        self._id = _id
        self.fitness = fitness
    
    def set_id(self, _id):
        self._id = _id
    def get_id(self):
        return self._id