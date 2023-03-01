from abc import ABCMeta, abstractmethod

class IProblem:
    __metaclass__ = ABCMeta

    @abstractmethod
    def objective_function(self, point):
        raise NotImplementedError

    @abstractmethod
    def preprocess_function(self, point):
        raise NotImplementedError

    @abstractmethod
    def repair_function(self, point):
        raise NotImplementedError
    