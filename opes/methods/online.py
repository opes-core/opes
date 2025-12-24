import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opes.methods.base_optimizer import Optimizer
from ..utils import extract_trim, find_regularizer, test_integrity, find_constraint
from ..errors import OptimizationError, PortfolioError

class BCRP(Optimizer):
    """
    Best Constant Rebalanced Portfolio (BCRP) optimizer.

    Finds the static weight vector that would have maximized the total wealth 
    of the portfolio in hindsight over the provided historical data.
    """
    def __init__(self, reg=None, strength=1):
        """
        Initializes the BCRP optimizer.

        :param reg: A regularization function or name.
        :param strength: Scalar multiplier for the regularization penalty.
        """
        self.identity = "bcrp"
        self.reg = find_regularizer(reg)
        self.strength = strength

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, w):
        """
        Processes data to extract tickers and return history, and initializes weights.

        :param data: Input financial data.
        :param w: Initial weights.
        :return: Cleaned return data array.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights)
        return data
    
    def optimize(self, data=None, w=None):
        """
        Executes the BCRP optimization by maximizing the product of portfolio returns.

        :param data: Historical return data.
        :param w: Initial weight vector.
        :return: Optimized weight vector (the constant rebalancing vector).
        :raises OptimizationError: If the optimization solver fails.
        """
        # Preparing optimization and finding constraint
        # Bounds are defaulted to (0,1), constrained to the simplex
        trimmed_return_data = self.prepare_optimization_inputs(data, w)
        constraint = find_constraint(bounds=(0,1))
        w = self.weights
        
        # Optimization objective and results
        def f(w):
            X = np.prod((1 + trimmed_return_data) @ w)
            return -X + self.strength * self.reg(w)
        result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': constraint}])
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(f"BCRP optimization failed: {result.message}")

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

class ExponentialGradient(Optimizer):
    """
    Exponential Gradient (EG) optimizer for online portfolio selection.

    An iterative update strategy that adjusts weights based on the relative 
    performance of assets in the most recent period, used to track the best 
    performing assets over time.
    """
    def __init__(self, learning_rate=0.3):
        """
        Initializes the Exponential Gradient optimizer.

        :param learning_rate: Scalar step size (eta) for the weight update rule.
        """
        self.identity = "expgrad"
        self.learning_rate = learning_rate

        self.tickers = None
        self.weights = None
    
    def prepare_inputs(self, data, w):
        """
        Extracts recent returns and initializes the weight vector for the update step.

        :param data: Input financial data.
        :param w: Current/Previous weights to be updated.
        :return: Cleaned return data array.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        # Defaulting previous weights to 1/N if none are given
        self.tickers, data = extract_trim(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights)
        return data
    
    def optimize(self, data=None, w=None):
        """
        Performs the Exponential Gradient update step.

        Uses the log-sum-exp technique to update weights based on the gradient 
        of the log-return relative to the most recent data point.

        :param data: Input data containing at least the most recent return.
        :param w: The current weight vector to be updated.
        :return: The newly updated weight vector.
        """
        # Preparing optimization and finding constraint
        # EG uses weight update method, so it takes the most recent return and uses it to update
        recent_return = self.prepare_inputs(data, w)[-1]
        portfolio_return = recent_return @ self.weights

        # Exponential Gradient update & normalization
        # We apply the log-sum-exp technique with subtracting the maximum to improve numerical stability
        # Weights are shift-invariant since they are exponentiated
        log_w = np.log(self.weights) + self.learning_rate * recent_return / portfolio_return
        log_w -= log_w.max()
        new_weights = np.exp(log_w)
        new_weights /= new_weights.sum()

        return self.weights