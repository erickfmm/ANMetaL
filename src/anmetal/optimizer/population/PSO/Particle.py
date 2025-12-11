from anmetal.optimizer.population.ISolution import SolutionWithId


class SolutionParticle(SolutionWithId):
    def __init__(self, _id, point, fitness, velocity):
        self.point = point
        self._id = _id
        self.fitness = fitness
        self.velocity = velocity
        self.set_best_point(point, fitness)
    
    def set_best_point(self, newpoint, newfitness):
        self.best_point = [x for x in newpoint]
        self.best_fitness = newfitness
    
    def set_velocity(self, _velocity):
        self.velocity = _velocity
    def get_velocity(self):
        return self.velocity

    def set_id(self, _id):
        self._id = _id
    def get_id(self):
        return self._id