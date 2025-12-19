import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from base_optimizer import Optimizer
from ..utils import trimmer, find_regulizer, test_integrity
from ..errors import PortfolioError, DataError, OptimizationError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Adding handler if no handlers exist
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", "%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class MaxMean(Optimizer):
    def __init__(self, reg=None, strength=1, mean=None):
        self.reg = find_regulizer(reg)
        self.strength = strength
        self.mean = mean
        self.tickers = None
        self.weights = None
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        
        # Extracting trimmed return data from OHLCV and obtaining tickers
        data = trimmer(data)
        self.tickers =  data.columns.get_level_values(0).unique().tolist()
        
        # Checking for mean and weights and assigning optimization data accordingly
        if self.mean is None:
            self.mean = data.mean()
        self.mean = np.array(self.mean)
        if w is None:
            w = np.ones(len(self.tickers))
        w = np.array(self.weights)

        # Testing data integrity function. Raises an error if the data is not the required format
        test_integrity(tickers=self.tickers, weights=w, mean=self.mean, bounds=weight_bounds)
        
        # Optimization objective and results
        def f(w):
            return -(self.mean @ w - self.strength * self.reg(w))
        result = minimize(f, w, method='SLSQP', bounds=[weight_bounds]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}]) # Construct constraints function
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError("Maximum Mean Optimization failed")

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