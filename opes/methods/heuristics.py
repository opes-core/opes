import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opes.methods.base_optimizer import Optimizer
from ..utils import extract_trim, find_regularizer, test_integrity, find_constraint
from ..errors import OptimizationError, PortfolioError, DataError

class Uniform(Optimizer):
    """
    Equal-weighted (1/N) portfolio optimizer.

    Assigns an identical weight to every asset in the portfolio regardless of 
    performance or risk metrics.
    """
    def __init__(self):
        """
        Initializes the Uniform optimizer.
        """
        self.identity = "uniform"

        self.tickers = None
        self.weights = None
    
    def extract_tickers(self, data):
        """
        Extracts unique ticker symbols from the columns of the input DataFrame.

        :param data: Input DataFrame with multi-index columns.
        :raises DataError: If the provided data is None.
        """
        if data is None:
            raise DataError("Portfolio data not specified.")
        
        # Extracting tickers
        self.tickers = data.columns.get_level_values(0).unique().tolist()
    
    def optimize(self, data=None):
        """
        Assigns equal weights to all assets.

        :param data: Input portfolio data (OHLCV).
        :return: Numpy array of equal weights.
        """
        self.extract_tickers(data)

        # Assigning weights and returning the same
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers))
        return self.weights

class InverseVolatility(Optimizer):
    """
    Inverse Volatility weighted portfolio optimizer.

    Allocates weights proportional to the reciprocal of each asset's standard deviation. 
    Assets with lower volatility receive higher allocations.
    """
    def __init__(self):
        """
        Initializes the InverseVolatility optimizer.
        """
        self.identity = "invvol"
        self.volarray = None

        self.tickers = None
        self.weights = None
    
    def prepare_inputs(self, data):
        """
        Processes data to extract tickers and calculate the volatility (standard deviation) array.

        :param data: Input OHLCV data grouped by ticker.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)

        # Extracting volatility array and testing integrity
        self.volarray = np.std(data, axis=0, ddof=1)
        test_integrity(tickers=self.tickers, volatility_array=self.volarray)
    
    def optimize(self, data=None):
        """
        Calculates weights based on inverse volatility.

        :param data: Input portfolio data.
        :return: Numpy array of weights normalized to sum to 1.
        """
        # Preparing inputs for finding weights
        self.prepare_inputs(data)
        self.weights = (1 / self.volarray) / ( 1 / self.volarray).sum()
        return self.weights

class SoftmaxMean(Optimizer):
    """
    Softmax Mean portfolio optimizer.

    Allocates weights by applying the softmax function to the mean returns of assets, 
    scaled by a temperature parameter.
    """
    def __init__(self, temperature=1):
        """
        Initializes the SoftmaxMean optimizer.

        :param temperature: Scalar that controls the sensitivity of the weights to return 
                            differences (higher temperature leads to more uniform weights).
        """
        self.identity = "softmean"
        self.mean = None
        self.temperature = temperature

        self.tickers = None
        self.weights = None
    
    def prepare_inputs(self, data, custom_mean=None):
        """
        Processes data to extract tickers and calculate mean returns.

        :param data: Input OHLCV data grouped by ticker.
        :param custom_mean: Custom mean vector
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)

        # Quick check for temperature validity
        if self.temperature <= 0:
            raise PortfolioError(f"Invalid temperature. Expected within bounds (0, inf], got {self.temperature})")

        # Extracting mean and testing integrity
        self.mean = np.mean(data, axis=0) if custom_mean is None else custom_mean
        test_integrity(tickers=self.tickers, mean=self.mean)
    
    def optimize(self, data=None, custom_mean=None):
        """
        Calculates weights using the softmax transformation of mean returns.

        :param data: Input portfolio data.
        :param custom_mean: Custom mean vector
        :return: Numpy array of softmax-transformed weights.
        """
        # Preparing inputs
        self.prepare_inputs(data, custom_mean=custom_mean)
    
        # Solving weights
        self.weights = np.exp(self.mean / self.temperature - np.max(self.mean / self.temperature))
        self.weights /= self.weights.sum()
        return self.weights

class MaxDiversification(Optimizer):
    """
    Maximum Diversification Ratio optimizer.

    Maximizes the ratio of the weighted average of asset volatilities to the 
    total portfolio volatility (the Diversification Ratio).
    """
    def __init__(self, reg=None, strength=0):
        """
        Initializes the MaxDiversification optimizer.

        :param reg: A regularization function or name.
        :param strength: Scalar multiplier for the regularization penalty.
        """
        self.identity = "maxdiverse"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.covariance = None
        self.volarray = None

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w, custom_cov=None):
        """
        Extracts returns and calculates the covariance matrix and per-asset volatility.

        :param data: Input OHLCV data grouped by ticker.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weights.
        :param custom_cov: Custom covariance from the user.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)
    
        # Checking for covariance, per-asset volatility and weights
        if custom_cov is None:
            # Handling invertibility using the small epsilon * identity matrix
            # small epsilon scales with the trace of the covariance
            self.covariance = np.cov(data, rowvar=False)
            epsilon = 1e-3 * np.trace(self.covariance) / self.covariance.shape[0]
            self.covariance =  self.covariance + epsilon * np.eye(self.covariance.shape[0])
        else:
            self.covariance = custom_cov
        self.volarray = np.sqrt(np.diag(self.covariance))   
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity
        test_integrity(tickers=self.tickers, weights=self.weights, cov=self.covariance, bounds=weight_bounds, volatility_array=self.volarray)
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None, custom_cov=None):
        """
        Executes the Maximum Diversification optimization.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for asset weights.
        :param w: Initial weight vector.
        :return: Optimized weight vector.
        :raises OptimizationError: If the SLSQP solver fails.
        :param custom_cov: Custom covariance matrix from the user.
        """
        # Preparing optimization and finding constraint
        self.prepare_optimization_inputs(data, weight_bounds, w, custom_cov=custom_cov)
        constraint = find_constraint(weight_bounds)
        w = self.weights
        
        # Optimization objective and results
        def f(w):
            var = w @ self.covariance @ w
            weightvol = w @ self.volarray
            return -(weightvol / np.sqrt(var)) + self.strength * self.reg(w)
        result = minimize(f, w, method='SLSQP', bounds=[weight_bounds]*len(w), constraints=constraint)
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(f"Maximum diversification optimization failed: {result.message}")

    def set_regularizer(self, reg=None, strength=1):
        """
        Updates the regularization function and its penalty strength.

        :param reg: The regularization function or name (e.g., 'l1', 'l2') to apply.
        :param strength: Scalar multiplier for the regularization penalty.
        """
        self.reg = find_regularizer(reg)
        self.strength = strength

class RiskParity(Optimizer):
    """
    Equal Risk Contribution (Risk Parity) optimizer.

    Calculates weights such that each asset contributes an equal amount of 
    marginal risk to the total portfolio volatility.
    """
    def __init__(self, reg=None, strength=0):
        """
        Initializes the RiskParity optimizer.

        :param reg: A regularization function or name.
        :param strength: Scalar multiplier for the regularization penalty.
        """
        self.identity = "riskparity"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.covariance = None

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w, custom_cov=None):
        """
        Processes data and calculates the covariance matrix for risk contribution analysis.

        :param data: Input OHLCV or return data.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weights.
        :param custom_cov: Custom covariance from the user.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)
    
        # Checking for covariance and weights
        if custom_cov is None:
            # Handling invertibility using the small epsilon * identity matrix
            # small epsilon scales with the trace of the covariance
            self.covariance = np.cov(data, rowvar=False)
            epsilon = 1e-3 * np.trace(self.covariance) / self.covariance.shape[0]
            self.covariance =  self.covariance + epsilon * np.eye(self.covariance.shape[0])
        else:
            self.covariance = custom_cov
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)

        # Functions to test data integrity
        test_integrity(tickers=self.tickers, weights=self.weights, cov=self.covariance, bounds=weight_bounds)
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None, custom_cov=None):
        """
        Executes the Risk Parity optimization by minimizing the variance of risk contributions.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for asset weights.
        :param w: Initial weight vector.
        :return: Optimized weight vector.
        :raises OptimizationError: If the optimization fails to converge.
        :param custom_cov: Custom covariance matrix from the user.
        """
        # Preparing optimization and finding constraint
        self.prepare_optimization_inputs(data, weight_bounds, w, custom_cov=custom_cov)
        constraint = find_constraint(weight_bounds)
        w = self.weights
        
        # Optimization objective and results
        def f(w):
            portfolio_volatility = max(np.sqrt((w @ self.covariance @ w)), 1e-10)
            risk_contribution = w * (self.covariance @ w) / portfolio_volatility
            target_contribution = portfolio_volatility / len(w)
            return np.sum((risk_contribution - target_contribution)**2) + self.strength * self.reg(w)
        result = minimize(f, w, method='SLSQP', bounds=[weight_bounds]*len(w), constraints=constraint)
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(f"Risk parity optimization failed: {result.message}")

    def set_regularizer(self, reg=None, strength=1):
        """
        Updates the regularization function and its penalty strength.

        :param reg: The regularization function or name (e.g., 'l1', 'l2') to apply.
        :param strength: Scalar multiplier for the regularization penalty.
        """
        self.reg = find_regularizer(reg)
        self.strength = strength

class REPO(Optimizer):
    """
    Return-Entropy Portfolio Optimizer (REPO).

    Optimizes the portfolio by balancing expected returns against the Shannon entropy 
    of the portfolio's return distribution. Higher entropy implies a more 
    uncertain/spread-out return distribution.
    """
    def __init__(self, risk_aversion=1, reg=None, strength=1):
        """
        Initializes the REPO optimizer.

        :param risk_aversion: Scalar weighting the importance of entropy (risk) relative to mean return.
        :param reg: A regularization function or name.
        :param strength: Scalar multiplier for the regularization penalty.
        """
        self.identity = "repo"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.risk_aversion = risk_aversion
        self.mean = None

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w, bins, custom_mean=None):
        """
        Processes input data, calculates mean returns, and validates histogram-based parameters.

        :param data: Input OHLCV or return data.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weight vector.
        :param bins: Number of bins for the return distribution histogram.
        :param custom_mean: Custom mean vector
        :return: Cleaned return data array.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)
        self.mean = np.mean(data, axis=0) if custom_mean is None else custom_mean
        
        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds, mean=self.mean, hist_bins=bins)
        return data
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None, bin=20, custom_mean=None):
        """
        Executes the Return-Entropy optimization.

        Minimizes the objective: -Mean + (risk_aversion * Entropy) + Penalty.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for asset weights.
        :param w: Initial weight vector.
        :param bin: Number of bins used to estimate the return probability distribution.
        :param custom_mean: Custom mean vector
        :return: Optimized weight vector.
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self.prepare_optimization_inputs(data, weight_bounds, w, bin, custom_mean=custom_mean)
        constraint = find_constraint(weight_bounds)
        w = self.weights
        
        # Optimization objective and results
        def f(w):
            X = trimmed_return_data @ w
            # Constructing histogram
            counts = np.histogram(X, bins=bin, density=False)[0]
            probabilities = counts / counts.sum()
            probabilities = probabilities[probabilities > 0]
            # Computing both the terms
            mean_term = self.mean @ w
            entropy_term = -np.sum(probabilities * np.log(probabilities))
            return -mean_term + self.risk_aversion * entropy_term + self.strength * self.reg(w)
        result = minimize(f, w, method='SLSQP', bounds=[weight_bounds]*len(w), constraints=constraint)
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(f"REPO optimization failed: {result.message}")

    def set_regularizer(self, reg=None, strength=1):
        """
        Updates the regularization function and its penalty strength.

        :param reg: The regularization function or name (e.g., 'l1', 'l2') to apply.
        :param strength: Scalar multiplier for the regularization penalty.
        """
        self.reg = find_regularizer(reg)
        self.strength = strength

class DEPO(Optimizer):
    """
    Discrete-Entropy Portfolio Optimizer (DEPO).

    Combines Kelly Criterion (logarithmic wealth maximization) with a penalty 
    based on the Kullback-Leibler (KL) divergence between the portfolio's return 
    distribution and a uniform distribution.
    """
    def __init__(self, risk_aversion=1, reg=None, strength=1):
        """
        Initializes the DEPO optimizer.

        :param risk_aversion: Scalar weighting applied to the KL divergence penalty.
        :param reg: A regularization function or name.
        :param strength: Scalar multiplier for the regularization penalty.
        """
        self.identity = "depo"
        self.reg = find_regularizer(reg)
        self.strength = strength
        self.risk_aversion = risk_aversion

        self.tickers = None
        self.weights = None
    
    def prepare_optimization_inputs(self, data, weight_bounds, w, bins):
        """
        Processes data and validates integrity for divergence-based optimization.

        :param data: Input OHLCV or return data.
        :param weight_bounds: Tuple of (min_weight, max_weight).
        :param w: Initial weight vector.
        :param bins: Number of bins for the return distribution histogram.
        :return: Cleaned return data array.
        """
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers) if w is None else w, dtype=float)
        
        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds, hist_bins=bins)
        return data
    
    def optimize(self, data=None, weight_bounds=(0,1), w=None, hist_bin=20):
        """
        Executes the Discrete-Entropy optimization.

        Minimizes the objective: -Expected_Log_Return + (risk_aversion * KL_Divergence) + Penalty.

        :param data: Input data for optimization.
        :param weight_bounds: Boundary constraints for asset weights.
        :param w: Initial weight vector.
        :param hist_bin: Number of bins used to estimate the probability distribution.
        :return: Optimized weight vector.
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self.prepare_optimization_inputs(data, weight_bounds, w, hist_bin)
        constraint = find_constraint(weight_bounds)
        w = self.weights
        
        # Optimization objective and results
        def f(w):
            X = trimmed_return_data @ w
            # Constructing histogram
            counts = np.histogram(X, bins=hist_bin, density=False)[0]
            probabilities = counts / counts.sum()
            probabilities = probabilities[probabilities > 0]
            uniform_prob = np.ones(len(probabilities)) / len(probabilities)
            # Computing both the terms
            X = np.maximum(X, -0.99)
            kullback_leibler_penalty = np.sum(probabilities * np.log(probabilities / uniform_prob))
            return -np.mean(np.log(1 + X)) + self.risk_aversion * kullback_leibler_penalty + self.strength * self.reg(w)
        result = minimize(f, w, method='SLSQP', bounds=[weight_bounds]*len(w), constraints=constraint)
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(f"DEPO optimization failed: {result.message}")

    def set_regularizer(self, reg=None, strength=1):
        """
        Updates the regularization function and its penalty strength.

        :param reg: The regularization function or name (e.g., 'l1', 'l2') to apply.
        :param strength: Scalar multiplier for the regularization penalty.
        """
        self.reg = find_regularizer(reg)
        self.strength = strength