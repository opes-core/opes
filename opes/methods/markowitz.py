import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opes.methods.base_optimizer import Optimizer
from ..utils import extract_trim, find_regularizer, test_integrity, find_constraint
from ..errors import PortfolioError, DataError, OptimizationError

class MaxMean(Optimizer):


    def __init__(self, reg=None, strength=1, mean=None):
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.mean = mean
        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)
        
        # Checking for mean and weights and assigning optimization data accordingly
        self.mean = np.array(data.mean(axis=1) if self.mean is None else self.mean, dtype=float)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, mean=self.mean, bounds=weight_bounds)
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        
        # Preparing optimization and finding constraint
        self.prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights
        
        # Optimization objective and results
        def f(w):
            return -(self.mean @ w - self.strength * self.reg(w))
        result = minimize(f, w, method='SLSQP', bounds=[weight_bounds]*len(w), constraints= [{'type':'eq','fun': constraint}])
        if result.success:
            self.weights = np.where(result.x > 1e-8, result.x, 0)
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