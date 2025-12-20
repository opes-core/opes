import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opes.methods.base_optimizer import Optimizer
from ..utils import extract_trim, find_regularizer, test_integrity, find_constraint
from ..errors import PortfolioError, DataError, OptimizationError

class MaxMean(Optimizer):


    def __init__(self, reg=None, strength=1):
        self.identity = "maxmean"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.mean = None

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)
        
        # Checking for mean and weights and assigning optimization data accordingly
        self.mean = np.mean(data, axis=1)
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
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError("Maximum mean optimization failed")

class MinVariance(Optimizer):


    def __init__(self, reg=None, strength=1):
        self.identity = "gmv"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.covariance = None

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)
    
        # Checking for covariance and weights and assigning optimization data accordingly
        self.covariance = np.cov(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, cov=self.covariance, bounds=weight_bounds)
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        
        # Preparing optimization and finding constraint
        self.prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights
        
        # Optimization objective and results
        def f(w):
            return w @ self.covariance @ w + self.strength * self.reg(w)
        result = minimize(f, w, method='SLSQP', bounds=[weight_bounds]*len(w), constraints= [{'type':'eq','fun': constraint}])
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError("Global minimum optimization failed")

class MeanVariance(Optimizer):


    def __init__(self, risk_aversion=0.5, reg=None, strength=1):
        self.identity = "mvo"
        self.reg = find_regularizer(reg)
        self.risk_aversion = risk_aversion
        self.strength = strength
        self.covariance = None
        self.mean = None

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)
    
        # Checking for mean, covaraince and weights and assigning optimization data accordingly
        self.mean = np.mean(data, axis=1)
        self.covariance = np.cov(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, cov=self.covariance, bounds=weight_bounds)
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        
        # Preparing optimization and finding constraint
        self.prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights
        
        # Optimization objective and results
        def f(w):
            return -self.mean @ w + (self.risk_aversion / 2) *(w @ self.covariance @ w) + self.strength * self.reg(w)
        result = minimize(f, w, method='SLSQP', bounds=[weight_bounds]*len(w), constraints= [{'type':'eq','fun': constraint}])
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError("Mean variance optimization failed")

class MaxSharpe(Optimizer):


    def __init__(self, risk_free=0.01, reg=None, strength=1):
        self.identity = "maxsharpe"
        self.reg = find_regularizer(reg)
        self.risk_free = risk_free
        self.strength = strength
        self.covariance = None
        self.mean = None

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)
    
        # Checking for mean, covariance and weights and assigning optimization data accordingly
        self.mean = np.mean(data, axis=1)
        self.covariance = np.cov(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, cov=self.covariance, bounds=weight_bounds)
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        
        # Preparing optimization and finding constraint
        self.prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights
        
        # Optimization objective and results
        def f(w):
            return - ((self.mean @ w - self.risk_free) /  max(np.sqrt((w @ self.covariance @ w)), 1e-10) - self.strength * self.reg(w))
        result = minimize(f, w, method='SLSQP', bounds=[weight_bounds]*len(w), constraints= [{'type':'eq','fun': constraint}])
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError("Maximum sharpe optimization failed")