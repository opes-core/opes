import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opes.methods.base_optimizer import Optimizer
from ..utils import extract_trim, find_regularizer, test_integrity, find_constraint
from ..errors import OptimizationError, PortfolioError

class CVaR(Optimizer):
    """
    Optimizer for minimizing the Conditional Value at Risk (CVaR), also known as Expected Shortfall.

    This class minimizes the tail risk of the portfolio at a specific confidence level 
    using the Rockafellar-Uryasev linear programming formulation.
    """
    def __init__(self, confidence=0.95, reg=None, strength=0):
        """
        Initializes the CVaR optimizer.

        :param confidence: The confidence level (alpha) for the VaR/CVaR calculation (e.g., 0.95).
        :param reg: A regularization function or name.
        :param strength: Scalar multiplier for the regularization penalty.
        """
        self.identity = "cvar"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.alpha = confidence

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        """
        Prepares returns, tickers, and initial weights while validating CVaR parameters.

        :param data: Input OHLCV data grouped by ticker.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weights.
        :return: Trimmed return data array.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds, confidence=self.alpha)
        return data
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        """
        Executes CVaR optimization by minimizing the auxiliary objective function 
        including Value-at-Risk as a decision variable.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for asset weights.
        :param w: Initial weights vector.
        :return: Optimized weight vector.
        :raises OptimizationError: If the SLSQP solver fails to converge.
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self.prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds, constraint_type=2)
        w = self.weights
        
        # Optimization objective and results
        # Appending initial VaR value, 1, to parameter array
        param_array = np.append(w, 1)
        def f(x):
            w, v = x[:-1], x[-1]
            X = -trimmed_return_data @ w
            excess = np.mean(np.maximum(X - v, 0.0))
            return (v + excess / (1 - self.alpha) + self.strength * self.reg(w))
        result = minimize(f, param_array, method='SLSQP', bounds=[weight_bounds]*len(w) + [(None,None)], constraints= [{'type':'eq','fun': constraint}])
        if result.success:
            self.weights = result.x[:-1]
            return self.weights
        else:
            raise OptimizationError(f"CVaR optimization failed: {result.message}")

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

class MeanCVaR(Optimizer):
    """
    Optimizer for Mean-CVaR portfolio efficiency.

    Solves the trade-off: min(γ * CVaR - (1-γ) * Mean), where γ represents 
    the risk aversion toward tail events.
    """
    def __init__(self, risk_aversion=0.5, confidence=0.95, reg=None, strength=0):
        """
        Initializes the Mean-CVaR optimizer.

        :param risk_aversion: Weight applied to the CVaR component vs the mean return.
        :param confidence: The confidence level (alpha) for CVaR calculation.
        :param reg: Regularization function.
        :param strength: Regularization strength.
        """
        self.identity = "mcvar"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.alpha = confidence
        self.risk_aversion = risk_aversion
        self.mean = None

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w, custom_mean=None):
        """
        Extracts returns, calculates mean return vector, and validates CVaR parameters.

        :param data: Input OHLCV data grouped by ticker.
        :param weight_bounds: Weight constraints.
        :param w: Initial weights.
        :param custom_mean: Custom mean vector
        :return: Cleaned return data array.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.mean = np.mean(data, axis=0) if custom_mean is None else custom_mean
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, mean=self.mean, bounds=weight_bounds, confidence=self.alpha)
        return data
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None, custom_mean=None):
        """
        Executes Mean-CVaR optimization.
        
        Minimizes the objective: (risk_aversion * CVaR) - mean_return + penalty.

        :param data: Input optimization data.
        :param weight_bounds: Weight constraints.
        :param w: Initial weights.
        :param custom_mean: Custom mean vector
        :return: Optimized weight vector.
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self.prepare_optimization_inputs(data, weight_bounds, w, custom_mean=custom_mean)
        constraint = find_constraint(weight_bounds, constraint_type=2)
        w = self.weights
        
        # Optimization objective and results
        # Appending initial VaR value, 1, to parameter array
        param_array = np.append(w, 1)
        def f(x):
            w, v = x[:-1], x[-1]
            X = -trimmed_return_data @ w
            excess = np.mean(np.maximum(X - v, 0.0))
            mean  = self.mean @ w
            return (self.risk_aversion * (v + excess / (1 - self.alpha)) + self.strength * self.reg(w) - mean)
        result = minimize(f, param_array, method='SLSQP', bounds=[weight_bounds]*len(w) + [(None,None)], constraints= [{'type':'eq','fun': constraint}])
        if result.success:
            self.weights = result.x[:-1]
            return self.weights
        else:
            raise OptimizationError(f"Mean CVaR optimization failed: {result.message}")

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

class EVaR(Optimizer):
    """
    Optimizer for minimizing Entropic Value at Risk (EVaR).

    EVaR is a coherent risk measure derived from the Chernoff bound, providing 
    a tighter upper bound for Value at Risk and CVaR.
    """
    def __init__(self, confidence=0.85, reg=None, strength=0):
        """
        Initializes the EVaR optimizer.

        :param confidence: The confidence level (alpha) for the entropic risk measure.
        :param reg: Regularization function.
        :param strength: Regularization strength.
        """
        self.identity = "evar"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.alpha = confidence

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        """
        Extracts tickers and return data for EVaR processing.

        :param data: Input OHLCV data grouped by ticker.
        :param weight_bounds: Weight constraints.
        :param w: Initial weights.
        :return: Cleaned return data array.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds, confidence=self.alpha)
        return data
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        """
        Executes Entropic Value at Risk optimization by minimizing the 
        logarithmic moment generating function of the portfolio returns.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for weights.
        :param w: Initial weights vector.
        :return: Optimized weight vector.
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self.prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds, constraint_type=2)
        w = self.weights
        
        # Optimization objective and results
        # Appending dual variable value as 1 to parameter array
        param_array = np.append(w, 1)
        def f(x):
            w,s = x[:-1], x[-1]
            X = trimmed_return_data @ w
            return (1/s) * (np.log(np.mean(np.exp(-s * X))) - np.log(1 - self.alpha)) + self.strength * self.reg(w)
        result = minimize(f, param_array, method='SLSQP', bounds=[weight_bounds]*len(w) + [(1e-8,None)], constraints= [{'type':'eq','fun': constraint}])
        if result.success:
            self.weights = result.x[:-1]
            return self.weights
        else:
            raise OptimizationError(f"EVaR optimization failed: {result.message}")

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

class MeanEVaR(Optimizer):
    """
    Optimizer for Mean-EVaR portfolio efficiency.

    Maximizes the return of the portfolio relative to its Entropic Value at Risk, 
    weighted by a risk aversion coefficient.
    """
    def __init__(self, risk_aversion=0.5, confidence=0.85, reg=None, strength=0):
        """
        Initializes the Mean-EVaR optimizer.

        :param risk_aversion: Scalar weighting the trade-off between mean and EVaR.
        :param confidence: The confidence level for the EVaR risk measure.
        :param reg: Regularization function.
        :param strength: Regularization strength.
        """
        self.identity = "mevar"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.alpha = confidence
        self.risk_aversion = risk_aversion
        self.mean = None

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w, custom_mean=None):
        """
        Processes data and calculates mean returns for the Mean-EVaR framework.

        :param data: Input OHLCV data grouped by ticker.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weights.
        :return: Cleaned return data array.
        :param custom_mean: Custom mean vector
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.mean = np.mean(data, axis=0) if custom_mean is None else custom_mean
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds, confidence=self.alpha)
        return data
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None, custom_mean=None):
        """
        Executes Mean-EVaR optimization.
        
        Minimizes: (risk_aversion * EVaR) - mean_return + penalty.

        :param data: Input optimization data.
        :param weight_bounds: Weight constraints.
        :param w: Initial weights.
        :param custom_mean: Custom mean vector
        :return: Optimized weight vector.
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self.prepare_optimization_inputs(data, weight_bounds, w, custom_mean=custom_mean)
        constraint = find_constraint(weight_bounds, constraint_type=2)
        w = self.weights
        
        # Optimization objective and results
        # Appending dual variable value as 1 to parameter array
        param_array = np.append(w, 1)
        def f(x):
            w,s = x[:-1], x[-1]
            X = trimmed_return_data @ w
            mean = self.mean @ w
            return self.risk_aversion * ((1/s) * (np.log(np.mean(np.exp(-s * X))) - np.log(1 - self.alpha))) + self.strength * self.reg(w) - mean
        result = minimize(f, param_array, method='SLSQP', bounds=[weight_bounds]*len(w) + [(1e-8,None)], constraints= [{'type':'eq','fun': constraint}])
        if result.success:
            self.weights = result.x[:-1]
            return self.weights
        else:
            raise OptimizationError(f"Mean EVaR optimization failed: {result.message}")

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

class EntropicRisk(Optimizer):
    """
    Optimizer for minimizing the Entropic Risk Measure (ERM).

    ERM is a risk measure derived from exponential utility, defined as 
    (1/γ) * log(E[exp(-γ * R)]), where γ is the risk aversion coefficient.
    """
    def __init__(self, risk_aversion=1, reg=None, strength=1):
        """
        Initializes the EntropicRisk optimizer.

        :param risk_aversion: The risk aversion coefficient (gamma). Must be non-zero.
        :param reg: A regularization function or name.
        :param strength: Scalar multiplier for the regularization penalty.
        """
        self.identity = "erm"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.risk_aversion = risk_aversion

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        """
        Processes input data, validates risk aversion bounds, and prepares weights.

        :param data: Input OHLCV or return data.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weights.
        :return: Cleaned return data array.
        :raises PortfolioError: If risk_aversion is set to zero.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Checking ERM risk aversion bounds
        if self.risk_aversion == 0:
            raise PortfolioError(f"Invalid ERM risk aversion. Expected within bounds (0, inf), Got {self.risk_aversion}")
        
        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds)
        return data
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        """
        Executes the Entropic Risk Measure optimization.

        Minimizes the convex entropic risk metric plus a regularization penalty.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for asset weights.
        :param w: Initial weight vector.
        :return: Optimized weight vector.
        :raises OptimizationError: If the SLSQP solver fails to converge.
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self.prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights
        
        # Optimization objective and results
        def f(w):
            X = trimmed_return_data @ w
            return 1/self.risk_aversion * np.log(np.mean(np.exp(-self.risk_aversion * X))) + self.strength * self.reg(w)
        result = minimize(f, w, method='SLSQP', bounds=[weight_bounds]*len(w), constraints= [{'type':'eq','fun': constraint}])
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(f"Entropic risk metric optimization failed: {result.message}")

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