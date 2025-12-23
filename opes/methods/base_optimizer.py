from abc import ABC, abstractmethod
import numpy as np
from opes.errors import PortfolioError

class Optimizer(ABC):
    """
    Abstract base class for portfolio optimization strategies.

    Defines the standard interface for calculating portfolio weights and 
    generating portfolio concentration statistics.
    """
    def __init__(self):
        """
        Initializes the optimizer with empty weights and tickers.
        """
        self.weights = None
        self.tickers = None
    
    @abstractmethod
    def optimize(self, data):
        """
        Abstract method to optimize portfolio weights based on the provided data.

        :param data: Input financial data (e.g., OHLCV or returns).
        :return: Optimized weight vector.
        """
        pass
    
    def stats(self):
        """
        Calculates and returns portfolio concentration and diversification statistics.

        Includes Portfolio Entropy, Herfindahl Index, and maximum weight allocation.

        :return: Dictionary containing tickers, weights, and concentration metrics.
        :raises PortfolioError: If weights have not been calculated via optimization.
        """
        if self.weights is None:
            raise PortfolioError("Weights not optimized")
        else:
            portfolio_entropy = -np.sum(np.abs(self.weights) * np.log(np.abs(self.weights) + 1e-12))
            herfindahl_index = np.sum(self.weights ** 2)
            max_weight = np.max(np.abs(self.weights))
            statistics = {
                "Tickers": self.tickers, 
                "Weights": np.round(self.weights, 5), 
                "Portfolio Entropy": portfolio_entropy, 
                "Herfindahl Index": herfindahl_index,
                "Absolute Max Weight" : max_weight
            }
            return statistics
    
    def clean_weights(self, threshold=1e-8):
        """
        Cleans the portfolio weights by setting very small positions to zero.

        Any weight whose absolute value is below the specified threshold is replaced with zero.
        This helps remove negligible allocations while keeping the array structure intact.

        :param threshold: Float specifying the minimum absolute weight to retain (default: 1e-8).
        :type threshold: float
        :raises PortfolioError: If weights have not been calculated or are None.
        """
        if self.weights is None:
            raise PortfolioError("Weights not optimized")
        else:
            self.weights[np.abs(self.weights) < threshold] = 0
            self.weights /= np.abs(self.weights).sum()
            return self.weights
