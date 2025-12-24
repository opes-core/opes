import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opes.methods.base_optimizer import Optimizer
from ..utils import extract_trim, find_regularizer, test_integrity, find_constraint
from ..errors import OptimizationError, PortfolioError

class Kelly(Optimizer):
    """
    Optimizer based on the Kelly Criterion for maximizing logarithmic growth.

    This class maximizes the expected value of the log of the portfolio return,
    incorporating a fractional Kelly component to manage volatility.
    """
    def __init__(self, fraction=1, reg=None, strength=1):
        """
        Initializes the Kelly optimizer.

        :param fraction: The Kelly fraction to apply (0 to 1).
        :param reg: A regularization function or name.
        :param strength: Scalar multiplier for the regularization penalty.
        """
        self.identity = "kelly"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.fraction = fraction

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        """
        Extracts tickers, prepares return data, and validates Kelly-specific parameters.

        :param data: Input OHLCV data grouped by ticker.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weights.
        :return: Trimmed return data array.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds, kelly_fraction=self.fraction)
        return data
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        """
        Executes the Kelly Criterion optimization using logarithmic utility.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for weights.
        :param w: Initial weights vector.
        :return: Optimized weight vector.
        :raises OptimizationError: If the solver fails to find a solution.
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self.prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights
        
        # Optimization objective and results
        def f(w):
            X = self.fraction * np.maximum((trimmed_return_data @ w), -0.99)
            return -np.mean(np.log(1 + X)) + self.strength * self.reg(w)
        result = minimize(f, w, method='SLSQP', bounds=[weight_bounds]*len(w), constraints=constraint)
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(f"Kelly criterion optimization failed: {result.message}")
    
    def set_regularizer(self, reg=None, strength=1):
        """
        Updates the regularization function and its penalty strength.

        :param reg: The regularization function or name (e.g., 'l1', 'l2') to apply.
        :param strength: Scalar multiplier for the regularization penalty.
        :raises PortfolioError: If no regularizer is provided.
        """
        if reg is None:
            raise PortfolioError("Regularizer not specified")
        self.reg = find_regularizer(reg)
        self.strength = strength

class QuadraticUtility(Optimizer):
    """
    Optimizer based on Quadratic Utility.

    Maximizes the expected utility function U(X) = E[X] - (γ/2) * E[X^2], 
    which serves as a second-order approximation to many utility functions.
    """
    def __init__(self, risk_aversion=0.5, reg=None, strength=1):
        """
        Initializes the Quadratic Utility optimizer.

        :param risk_aversion: Risk aversion coefficient (gamma).
        :param reg: Regularization function.
        :param strength: Regularization strength.
        """
        self.identity = "quadutil"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.risk_aversion = risk_aversion

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        """
        Processes data and validates parameters for the specific utility function.

        :param data: Input OHLCV data grouped by ticker.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weights.
        :return: Cleaned return data.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)
        
        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds)
        return data
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        """
        Executes the utility-based optimization.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for each asset weight.
        :param w: Initial weight vector.
        :return: Optimized weight vector.
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self.prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights
        
        # Optimization objective and results
        def f(w):
            X = 1 + np.maximum((trimmed_return_data @ w), -1)
            return np.mean(self.risk_aversion/2 * (X ** 2) - X) + self.strength * self.reg(w)
        result = minimize(f, w, method='SLSQP', bounds=[weight_bounds]*len(w), constraints=constraint)
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(f"Quadratic utility optimization failed: {result.message}")
    
    def set_regularizer(self, reg=None, strength=1):
        """
        Updates the regularization function and its penalty strength.

        :param reg: The regularization function or name (e.g., 'l1', 'l2') to apply.
        :param strength: Scalar multiplier for the regularization penalty.
        :raises PortfolioError: If no regularizer is provided.
        """
        if reg is None:
            raise PortfolioError("Regularizer not specified")
        self.reg = find_regularizer(reg)
        self.strength = strength

class CARA(Optimizer):
    """
    Optimizer based on Constant Absolute Risk Aversion (CARA).

    Maximizes Exponential Utility: U(r) = -E[exp(-γ * r)], where γ is the 
    coefficient of absolute risk aversion.
    """
    def __init__(self, risk_aversion=1, reg=None, strength=1):
        """
        Initializes the CARA optimizer.

        :param risk_aversion: Constant absolute risk aversion coefficient.
        :param reg: Regularization function.
        :param strength: Regularization strength.
        """
        self.identity = "cara"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.risk_aversion = risk_aversion

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        """
        Processes data and validates parameters for the specific utility function.

        :param data: Input OHLCV data grouped by ticker.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weights.
        :return: Cleaned return data.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds)
        return data
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        """
        Executes the utility-based optimization.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for each asset weight.
        :param w: Initial weight vector.
        :return: Optimized weight vector.
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self.prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights
        
        # Optimization objective and results
        def f(w):
            X = np.maximum((trimmed_return_data @ w), -1)
            return np.mean(np.exp( - self.risk_aversion * X)) + self.strength * self.reg(w)
        result = minimize(f, w, method='SLSQP', bounds=[weight_bounds]*len(w), constraints=constraint)
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(f"CARA optimization failed: {result.message}")

    def set_regularizer(self, reg=None, strength=1):
        """
        Updates the regularization function and its penalty strength.

        :param reg: The regularization function or name (e.g., 'l1', 'l2') to apply.
        :param strength: Scalar multiplier for the regularization penalty.
        :raises PortfolioError: If no regularizer is provided.
        """
        if reg is None:
            raise PortfolioError("Regularizer not specified")
        self.reg = find_regularizer(reg)
        self.strength = strength

class CRRA(Optimizer):
    """
    Optimizer based on Constant Relative Risk Aversion (CRRA).

    Maximizes Power Utility: U(r) = E[(1+r)^(1-γ) / (1-γ)], where γ is the 
    coefficient of relative risk aversion.
    """
    def __init__(self, risk_aversion=2, reg=None, strength=1):
        """
        Initializes the CRRA optimizer.

        :param risk_aversion: Relative risk aversion coefficient (must be > 1).
        :param reg: Regularization function.
        :param strength: Regularization strength.
        """
        self.identity = "crra"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.risk_aversion = risk_aversion

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        """
        Processes data and validates parameters for the specific utility function.

        :param data: Input OHLCV data grouped by ticker.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weights.
        :return: Cleaned return data.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Checking for CRRA risk aversion validity
        if self.risk_aversion <= 1:
            raise PortfolioError(f"CRRA risk aversion out of bounds. Expected within (1,inf), Got {self.risk_aversion}")
        
        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds)
        return data
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        """
        Executes the utility-based optimization.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for each asset weight.
        :param w: Initial weight vector.
        :return: Optimized weight vector.
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self.prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights
        
        # Optimization objective and results
        def f(w):
            X = np.maximum((trimmed_return_data @ w), -0.99)
            return -np.mean((1 + X) ** (1-self.risk_aversion)) / (1-self.risk_aversion) + self.strength * self.reg(w)
        result = minimize(f, w, method='SLSQP', bounds=[weight_bounds]*len(w), constraints=constraint)
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(f"CRRA optimization failed: {result.message}")

    def set_regularizer(self, reg=None, strength=1):
        """
        Updates the regularization function and its penalty strength.

        :param reg: The regularization function or name (e.g., 'l1', 'l2') to apply.
        :param strength: Scalar multiplier for the regularization penalty.
        :raises PortfolioError: If no regularizer is provided.
        """
        if reg is None:
            raise PortfolioError("Regularizer not specified")
        self.reg = find_regularizer(reg)
        self.strength = strength

class HARA(Optimizer):
    """
    Optimizer based on Hyperbolic Absolute Risk Aversion (HARA).

    A generalized utility framework that nests CRRA, CARA, and Quadratic utility.
    Objective: max E[(scale * (1+r) + shift)^(1-γ) / (1-γ)].
    """
    def __init__(self, risk_aversion=2, scale=1, shift=3, reg=None, strength=1):
        """
        Initializes the HARA optimizer.

        :param risk_aversion: Risk aversion parameter (gamma).
        :param scale: Scaling factor for the wealth/return term.
        :param shift: Shift parameter for the utility function.
        :param reg: Regularization function.
        :param strength: Regularization strength.
        """
        self.identity = "hara"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.risk_aversion = risk_aversion
        self.scale = scale
        self.shift = shift

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        """
        Processes data and validates parameters for the specific utility function.

        :param data: Input OHLCV data grouped by ticker.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weights.
        :return: Cleaned return data.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Checking for CRRA risk aversion validity
        if self.risk_aversion <= 1:
            raise PortfolioError(f"HARA risk aversion out of bounds. Expected within (1,inf), Got {self.risk_aversion}")
        if self.scale <= 0:
            raise PortfolioError(f"HARA scale value out of bounds. Expected within (0, inf), Got {self.scale}")
        
        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds)
        return data
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        """
        Executes the utility-based optimization.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for each asset weight.
        :param w: Initial weight vector.
        :return: Optimized weight vector.
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self.prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights
        
        # Optimization objective and results
        def f(w):
            X = np.maximum((trimmed_return_data @ w), -0.99)
            return - np.mean((self.scale * (1 + X) + self.shift) ** (1-self.risk_aversion)) / (1 - self.risk_aversion) + self.strength * self.reg(w)
        result = minimize(f, w, method='SLSQP', bounds=[weight_bounds]*len(w), constraints=constraint)
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(f"HARA optimization failed: {result.message}")

    def set_regularizer(self, reg=None, strength=1):
        """
        Updates the regularization function and its penalty strength.

        :param reg: The regularization function or name (e.g., 'l1', 'l2') to apply.
        :param strength: Scalar multiplier for the regularization penalty.
        :raises PortfolioError: If no regularizer is provided.
        """
        if reg is None:
            raise PortfolioError("Regularizer not specified")
        self.reg = find_regularizer(reg)
        self.strength = strength