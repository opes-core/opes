import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opes.methods.base_optimizer import Optimizer
from ..utils import extract_trim, find_regularizer, test_integrity, find_constraint
from ..errors import OptimizationError

class MaxMean(Optimizer):
    """
    Optimizer to maximize the portfolio's expected mean return.

    This class solves the objective: max(w'μ - λ * reg(w)), where μ is the 
    expected return vector and λ is the regularization strength.
    """
    def __init__(self, reg=None, strength=1):
        """
        Initializes the MaxMean optimizer.

        :param reg: A regularization function or name to apply to weights.
        :param strength: Scalar multiplier for the regularization penalty.
        """
        self.identity = "maxmean"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.mean = None

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        """
        Processes input data, extracts tickers, calculates mean returns, and validates integrity.

        :param data: Input OHLCV data grouped by ticker.
        :param weight_bounds: Tuple of (min_weight, max_weight) for optimization.
        :param w: Initial weights; if None, defaults to an equal-weighted portfolio.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)
        
        # Checking for mean and weights and assigning optimization data accordingly
        self.mean = np.mean(data, axis=0)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, mean=self.mean, bounds=weight_bounds)
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        """
        Executes the Maximum Mean optimization.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for each asset weight.
        :param w: Initial weight vector.
        :return: Optimized weight vector as a numpy array.
        :raises OptimizationError: If the SLSQP solver fails to converge.
        """
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
    """
    Optimizer for the Global Minimum Variance (GMV) portfolio.

    Solves the objective: min(w'Σw + λ * reg(w)), where Σ is the 
    covariance matrix.
    """
    def __init__(self, reg=None, strength=1):
        """
        Initializes the MinVariance optimizer.

        :param reg: A regularization function or name to apply to weights.
        :param strength: Scalar multiplier for the regularization penalty.
        """
        self.identity = "gmv"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.covariance = None

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        """
        Processes input data, calculates the covariance matrix, and validates integrity.

        :param data: Input OHLCV data grouped by ticker.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weights.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)
    
        # Checking for covariance and weights and assigning optimization data accordingly
        self.covariance = np.cov(data, rowvar=False)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, cov=self.covariance, bounds=weight_bounds)
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        """
        Executes the Global Minimum Variance optimization.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for each asset weight.
        :param w: Initial weight vector.
        :return: Optimized weight vector.
        :raises OptimizationError: If optimization fails.
        """
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
    """
    Standard Markowitz Mean-Variance Optimizer.

    Solves the objective: min(-w'μ + (γ/2) * w'Σw + λ * reg(w)), 
    where γ is the risk aversion coefficient.
    """

    def __init__(self, risk_aversion=0.5, reg=None, strength=1):
        """
        Initializes the Mean-Variance optimizer.

        :param risk_aversion: Scalar (gamma) controlling the trade-off between return and risk.
        :param reg: Regularization function.
        :param strength: Regularization strength.
        """
        self.identity = "mvo"
        self.reg = find_regularizer(reg)
        self.risk_aversion = risk_aversion
        self.strength = strength
        self.covariance = None
        self.mean = None

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        """
        Processes input data, calculates mean returns and covariance, and validates integrity.

        :param data: Input OHLCV data grouped by ticker.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weights.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)
    
        # Checking for mean, covaraince and weights and assigning optimization data accordingly
        self.mean = np.mean(data, axis=0)
        self.covariance = np.cov(data, rowvar=False)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, cov=self.covariance, bounds=weight_bounds)
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        """
        Executes the Mean-Variance optimization.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for each asset weight.
        :param w: Initial weight vector.
        :return: Optimized weight vector.
        """
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
    """
    Optimizer to maximize the portfolio Sharpe Ratio.

    Solves the objective: max((w'μ - Rf) / sqrt(w'Σw) - λ * reg(w)), 
    where Rf is the risk-free rate.
    """

    def __init__(self, risk_free=0.01, reg=None, strength=1):
        """
        Initializes the Max Sharpe optimizer.

        :param risk_free: The risk-free rate of return.
        :param reg: Regularization function.
        :param strength: Regularization strength.
        """
        self.identity = "maxsharpe"
        self.reg = find_regularizer(reg)
        self.risk_free = risk_free
        self.strength = strength
        self.covariance = None
        self.mean = None

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        """
        Processes input data, calculates mean returns and covariance, and validates integrity.

        :param data: Input OHLCV data grouped by ticker.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weights.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)
    
        # Checking for mean, covariance and weights and assigning optimization data accordingly
        self.mean = np.mean(data, axis=0)
        self.covariance = np.cov(data, rowvar=False)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, cov=self.covariance, bounds=weight_bounds)
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        """
        Executes the Maximum Sharpe Ratio optimization.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for each asset weight.
        :param w: Initial weight vector.
        :return: Optimized weight vector.
        """
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