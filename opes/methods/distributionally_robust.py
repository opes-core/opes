import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opes.methods.base_optimizer import Optimizer
from ..utils import extract_trim, test_integrity, find_constraint
from ..errors import OptimizationError, PortfolioError

class KLRobustMaxMean(Optimizer):
    """
    Distributionally Robust Maximum Mean Optimizer (Kullback-Leibler Ambiguity).

    Optimizes the expected return under the worst-case probability distribution 
    within a KL-divergence uncertainty ball (radius) around the empirical distribution.
    """
    def __init__(self, radius=0.2):
        """
        Initializes the Distributionally Robust Maximum Mean optimizer.

        :param radius: The size of the uncertainty set (KL-divergence radius). 
                       Larger values indicate higher uncertainty.
        """
        self.identity = "kldr-mm"
        self.radius = radius

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        """
        Processes data and validates the uncertainty radius for robust optimization.

        :param data: Input OHLCV or return data.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weight vector.
        :return: Cleaned return data array.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds, uncertainty_radius=self.radius)
        return data
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        """
        Executes the distributionally robust mean optimization.

        Uses a dual reformulation and the log-sum-exp technique to solve for 
        the worst-case expected return.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for asset weights.
        :param w: Initial weight vector.
        :return: Optimized weight vector.
        :raises OptimizationError: If the solver fails to converge.
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self.prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds, constraint_type=2)
        param_array = np.append(self.weights, 1)
        # Optimization objective and results
        def f(x):
            w, dual_var = x[:-1], x[-1]
            X = -(trimmed_return_data @ w / dual_var)
            # Utilize the log-sum-exp tecnique to ensure numerical stability
            m = np.max(X)
            return dual_var * self.radius + dual_var * (m + np.log(np.mean(np.exp(X - m))))
        result = minimize(f, param_array, method='SLSQP', bounds=[weight_bounds]*len(self.weights) + [(1e-3, None)], constraints=constraint)
        if result.success:
            self.weights = result.x[:-1]
            return self.weights
        else:
            raise OptimizationError(f"Distributionally robust maximum mean optimization failed: {result.message}")

class KLRobustKelly(Optimizer):
    """
    Distributionally Robust Kelly Criterion Optimizer (Kullback-Leibler Ambiguity).

    Maximizes the expected logarithmic wealth under the worst-case probability 
    distribution within a specified KL-divergence radius.
    """
    def __init__(self, fraction=1, radius=0.2):
        """
        Initializes the Distributionally Robust Kelly optimizer.

        :param fraction: The Kelly fraction (leverage) to apply.
        :param radius: The KL-divergence uncertainty radius.
        """
        self.identity = "kldr-kelly"
        self.radius = radius
        self.fraction = fraction

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w):
        """
        Extracts tickers and validates robustness parameters and weight constraints.

        :param data: Input OHLCV or return data.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weight vector.
        :return: Cleaned return data array.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds, uncertainty_radius=self.radius)
        return data
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None):
        """
        Executes the robust Kelly optimization.

        Solves the minimax problem of maximizing log-wealth over the worst-case 
        distribution using a dual variable approach.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for asset weights.
        :param w: Initial weight vector.
        :return: Optimized weight vector.
        :raises OptimizationError: If the optimization fails.
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self.prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds, constraint_type=2)
        param_array = np.append(self.weights, 1)
        # Optimization objective and results
        def f(x):
            w, dual_var = x[:-1], x[-1]
            E = np.mean(np.maximum((1 + self.fraction * (trimmed_return_data @ w)), 0.001) ** (-1 / dual_var))
            return dual_var * self.radius + dual_var * np.log(E)
        result = minimize(f, param_array, method='SLSQP', bounds=[weight_bounds]*len(self.weights) + [(1e-3, None)], constraints=constraint)
        if result.success:
            self.weights = result.x[:-1]
            return self.weights
        else:
            raise OptimizationError(f"Distributionally robust kelly criterion optimization failed: {result.message}")