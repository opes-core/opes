from abc import ABC, abstractmethod
import numpy as np
from opes.errors import PortfolioError

class Optimizer(ABC):
    """
    Optimizer template
    
    Contains:
    optimize(self, data) 
        Optimizes the portfolio and returns weights
    stats(self) 
        Returns detailed portfolio statistics
    """
    def __init__(self):
        self.weights = None
        self.tickers = None
    
    @abstractmethod
    def optimize(self, data):
        pass
    
    def stats(self):

        if self.weights is None:
            raise PortfolioError("Weights not optimized")
        else:
            portfolio_entropy = -np.sum(np.abs(self.weights) * np.log(np.abs(self.weights) + 1e-12))
            herfindahl_index = np.sum(self.weights ** 2)
            max_weight = np.max(np.abs(self.weights))
            statistics = {
                "Tickers": self.tickers, 
                "Weights": np.round(self.weights, 2), 
                "Portfolio Entropy": portfolio_entropy, 
                "Herfindahl Index": herfindahl_index,
                "Absolute Max Weight" : max_weight
            }
            return statistics