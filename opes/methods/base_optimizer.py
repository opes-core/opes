from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    Base optimizer template
    
    Contains:
    optimize(self, data) method which optimizes the portfolio and returns weights
    """
    @abstractmethod
    def optimize(self, data):
        pass