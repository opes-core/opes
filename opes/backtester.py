from numbers import Real
import time 
import inspect

import numpy as np
import pandas as pd
import scipy.stats as scistats
import matplotlib.pyplot as plt

from opes.errors import PortfolioError, DataError
from opes.utils import slippage, extract_trim

class Backtester():
    """
    A comprehensive backtesting engine for financial time series.

    This class manages training and testing datasets, ensuring that
    any missing values (NaNs) are removed for robust backtesting loops.
    It also stores transaction cost parameters for portfolio simulations.

    Attributes:
        train (pd.DataFrame): Training dataset with NaNs removed.
        test (pd.DataFrame): Testing dataset with NaNs removed.
        cost (dict): Dictionary specifying transaction cost parameters. 
                     Defaults to {'const': 10.0}.
    """
    def __init__(self, train_data=None, test_data=None, cost={'const': 10.0}):
        """
        Initializes the Backtester with optional training and testing data.

        Args:
            train_data (pd.DataFrame): Historical training data. 
                                                 Defaults to None.
            test_data (pd.DataFrame): Historical testing data. 
                                                Defaults to None.
            cost (dict, optional): Transaction cost parameters. Defaults to {'const': 10.0}.

        Notes:
            - NaN values in both train_data and test_data are automatically dropped 
              to prevent indexing errors during backtests.
            - The `cost` dictionary can be expanded to include other cost structures.
        """
        # Assigning by dropping nans to ensure proper indexing
        # Dropping nan rows results makes backtest loops robust and predictable
        self.train = train_data.dropna()
        self.test = test_data.dropna()
        self.cost = cost
    
    def backtest_integrity_check(self, optimizer, rebalance_freq, seed, cleanweights=False):
        """
        Validates inputs and configurations before running a backtest.

        This method performs a series of sanity checks to ensure that
        the backtesting process will not fail due to invalid data,
        optimizer objects, rebalance settings, random seeds, or cost models.

        Args:
            optimizer: Portfolio optimization object that must have an 'identity' attribute.
            rebalance_freq (int or None): Frequency of portfolio rebalancing in time steps.
                                           Must be a positive integer if specified.
            seed (int or None): Random seed for reproducibility. Must be an integer if specified.
            cleanweights (True or False): Filters infinitesmal weights and renormalizes

        Raises:
            DataError: If the training data length is insufficient for backtesting, or if the format doesn't match.
            PortfolioError: If the optimizer object, rebalance frequency, seed, or cost model is invalid.

        Validation checks include:
            - Training data length (minimum 5 rows, non-empty)
            - Training and test data format match
            - Optimizer object type and attribute presence
            - Rebalance frequency type and positivity
            - Random seed type
            - Cost model structure and parameters:
                * 'const': single real number
                * 'lognormal', 'gamma', 'inversegaussian': tuple/list of length 2
                * 'jump': tuple/list of length 3
        """
        # Checking train and test data length and format
        if len(self.train) < 5:
            raise DataError(f"Insufficient training data for backtest. Expected len(data) >= 5, got {len(self.train)}")
        if len(self.train) <= 0:
            raise DataError(f"Insufficient training data for backtest. Expected len(data) > 0, got {len(self.train)}")
        if not((self.train.columns.equals(self.test.columns))):
            raise DataError("Train and test DataFrames have different formats")
        if cleanweights is not True and cleanweights is not False:
            raise DataError(f"Invalid cleanweights variable. Expected True or False, got {cleanweights}")
        # Checking optimizer validity
        try:
            optimizer.identity
        except:
            raise PortfolioError(f"Portfolio object not given. Got {type(optimizer)}")
        # Checking rebalance frequency type and validity
        if rebalance_freq is not None:
            if rebalance_freq <= 0 or not isinstance(rebalance_freq, int):
                raise PortfolioError(f"Invalid rebalance frequency. Expected integer within bounds [1,T], Got {rebalance_freq}")
        # Validiating numpy seed
        if seed is not None and not isinstance(seed, int):
            raise PortfolioError(f"Invalid seed. Expected integer or None, Got {seed}")
        # Cost model validity - per model check
        if len(self.cost) != 1:
            raise PortfolioError(f"Invalid cost model. Cost model must be a dictionary of length 1, Got {len(self.cost)}")
        first_key = next(iter(self.cost))
        first_key_low = first_key.lower()
        if first_key_low not in ['const', 'lognormal', 'gamma', 'inversegaussian', 'jump']:
            raise PortfolioError(f"Unknown cost model: {first_key}")
        elif (first_key_low == 'const' and not isinstance(self.cost[first_key], Real)):
            raise PortfolioError(f"Unspecified cost value. Expected real number, got {type(self.cost[first_key])}")
        elif first_key_low in ['lognormal', 'gamma', 'inversegaussian'] and len(self.cost[first_key]) != 2:
            raise PortfolioError(f"Invalid cost model parameter length. Expected 2, got {len(self.cost[first_key])}")
        elif first_key_low == 'jump' and len(self.cost[first_key]) != 3:
            raise PortfolioError(f"Invalid jump cost model parameter length. Expected 3, got {len(self.cost[first_key])}")
            

    def backtest(self, optimizer, rebalance_freq=None, seed=None, weight_bounds=None, clean_weights=False):
        """
        Execute a portfolio backtest over the test dataset using a given optimizer.

        This method performs either a static-weight backtest or a rolling-weight
        backtest depending on whether `rebalance_freq` is specified. It also
        applies transaction costs and ensures no lookahead bias during rebalancing.
        For a rolling backtest, any common date values are dropped, the first occurrence
        is considered to be original and kept.

        Args:
            optimizer: Portfolio optimizer object with an `optimize` method.
            rebalance_freq (int or None): Frequency of rebalancing in time steps. 
                                           If None, a static weight backtest is performed.
            seed (int or None): Random seed for reproducible cost simulations.
            weight_bounds (optional): Bounds for portfolio weights passed to the optimizer 
                                      if supported.

        Returns:
            dict: Backtest results containing:
                - 'returns' (np.ndarray): Portfolio returns after accounting for costs.
                - 'weights' (np.ndarray): Portfolio weights at each timestep.
                - 'costs' (np.ndarray): Transaction costs applied at each timestep.

        Raises:
            DataError: If the optimizer does not accept weight bounds but `weight_bounds`
                       are provided.
            PortfolioError: If input validation fails (via `backtest_integrity_check`).

        Behavior:
            - Static weight backtest: Uses a single set of optimized weights for all test data.
            - Rolling weight backtest: Re-optimizes weights at intervals defined by `rebalance_freq`
              using only historical data up to the current point to prevent lookahead bias.
            - Transaction costs are applied using the `slippage` function, supporting various cost
              models (constant, lognormal, gamma, inversegaussian, jump).
            - Returns and weights are stored in arrays aligned with test data indices.
        """
        # Running backtester integrity checks
        self.backtest_integrity_check(optimizer, rebalance_freq, seed, cleanweights=clean_weights)
        # Backtest loop
        test_data = extract_trim(self.test)[1]
        # Static weight backtest
        if rebalance_freq is None:
            if weight_bounds is not None:
                if "weight_bounds" in inspect.signature(optimizer.optimize).parameters:
                    weights = optimizer.optimize(self.train, weight_bounds=weight_bounds)
                else:
                    raise DataError(f"Given portfolio strategy does not accept weight bounds")
            else:
                weights = optimizer.optimize(self.train)
            if clean_weights:
                weights = optimizer.clean_weights()
            weights_array = np.tile(weights, (len(test_data), 1))
        # Rolling weight backtest
        if rebalance_freq is not None:
            weights = [None] * len(test_data)
            if weight_bounds is not None:
                if "weight_bounds" in inspect.signature(optimizer.optimize).parameters:
                    temp_weights = optimizer.optimize(self.train, weight_bounds=weight_bounds)
                else:
                    raise DataError(f"Given portfolio strategy does not accept weight bounds")
            else:
                temp_weights = optimizer.optimize(self.train)
            if clean_weights:
                temp_weights = optimizer.clean_weights()
            weights[0] = temp_weights
            for t in range(1, len(test_data)):
                if t % rebalance_freq == 0:
                    # NO LOOKAHEAD BIAS
                    # Rebalance at timestep t using only past data (up to t, exclusive) to avoid lookahead bias
                    # Training data is pre-cleaned (no NaNs), test data up to t is also NaN-free
                    # Concatenating them preserves this property; dropna() handles edge cases safely
                    # The optimizer therefore only sees information available until the current decision point
                    if weight_bounds is not None:
                        if "weight_bounds" in inspect.signature(optimizer.optimize).parameters:
                            combined_dataset = pd.concat([self.train, self.test.iloc[:t]])
                            combined_dataset = combined_dataset[~combined_dataset.index.duplicated(keep="first")].dropna()
                            temp_weights = optimizer.optimize(combined_dataset, w=temp_weights, weight_bounds=weight_bounds)
                        else:
                            raise DataError(f"Given portfolio strategy does not accept weight bounds")
                    else:
                        combined_dataset = pd.concat([self.train, self.test.iloc[:t]])
                        combined_dataset = combined_dataset[~combined_dataset.index.duplicated(keep="first")].dropna()
                        temp_weights = optimizer.optimize(combined_dataset, w=temp_weights)
                    if clean_weights:
                        temp_weights = optimizer.clean_weights(temp_weights)
                weights[t] = temp_weights
            weights_array = np.vstack(weights)
        portfolio_returns = np.einsum('ij,ij->i', weights_array, test_data)
        costs_array = slippage(weights=weights_array, returns=portfolio_returns, cost=self.cost, numpy_seed=seed)
        portfolio_returns -= costs_array
        backtest_data = {
            'returns': portfolio_returns,
            'weights': weights_array,
            'costs': costs_array
        }
        return backtest_data

    def get_metrics(self, returns):
        """
        Computes a comprehensive set of portfolio performance metrics from returns.

        This method calculates risk-adjusted and absolute performance measures 
        commonly used in finance, including volatility, drawdowns, and tail risk metrics.

        Args:
            returns (array-like): Array or list of periodic portfolio returns.

        Returns:
            dict: Dictionary of performance metrics with the following keys:
                - 'sharpe': Sharpe ratio.
                - 'sortino': Sortino ratio.
                - 'volatility': Standard deviation of returns (%).
                - 'mean_return': Mean return (%).
                - 'total_return': Total cumulative return (%).
                - 'max_drawdown': Maximum drawdown.
                - 'var_95': Value at Risk at 95% confidence level.
                - 'cvar_95': Conditional Value at Risk (expected shortfall) at 95%.
                - 'skew': Skewness of returns.
                - 'kurtosis': Kurtosis of returns.
                - 'omega_0': Omega ratio (gain/loss ratio).

        Notes:
            - Volatility, mean, total return, and CAGR are scaled to percentages.
            - Tail risk metrics (VaR, CVaR) are based on the lower 5% of returns.
            - Returns should be cleaned (NaNs removed) before passing to this method.
        """
        # Caching repeated values
        returns = np.array(returns)
        downside_vol = returns[returns < 0].std()
        vol = returns.std()

        # Performance metrics
        AVERAGE = returns.mean()
        SHARPE = AVERAGE / vol if (vol > 0 and not np.isnan(vol)) else np.nan
        SORTINO = AVERAGE / downside_vol if (downside_vol > 0 and not np.isnan(downside_vol)) else np.nan
        VOLATILITY = vol if (vol > 0 or not np.isnan(vol)) else np.nan
        TOTAL = np.prod(1 + returns) - 1
        MAX_DD = np.max(1 - np.cumprod(1 + returns) / np.maximum.accumulate(np.cumprod(1 + returns)))
        VAR = -np.quantile(returns, 0.05)
        tail_returns = returns[returns <= -VAR]
        CVAR = -tail_returns.mean() if len(tail_returns) > 0 else np.nan
        SKEW = scistats.skew(returns)
        KURTOSIS = scistats.kurtosis(returns)
        OMEGA = np.sum(np.maximum(returns, 0)) / np.sum(np.maximum(-returns, 0))

        # Zipping Text and values
        performance_metrics = [
            'sharpe',
            'sortino',
            'volatility',
            'mean_return',
            'total_return',
            'max_drawdown',
            'var_95',
            'cvar_95',
            'skew',
            'kurtosis',
            'omega_0'
        ]
        values = [VOLATILITY, AVERAGE, TOTAL, MAX_DD, VAR, CVAR]
        results = [round(SHARPE, 5), round(SORTINO, 5)] + [round(x*100, 5) for x in values] + [round(SKEW, 5), round(KURTOSIS, 5), round(OMEGA, 5)]
        return dict(zip(performance_metrics, results))
    
    def plot_wealth(self, returns_dict, initial_wealth=1.0, savefig=False):
        """
        Plot the evolution of portfolio wealth over time on a logarithmic scale.

        This method visualizes cumulative wealth for one or multiple strategies 
        using their periodic returns. It also provides a breakeven reference line
        and optional saving of the plot to a file.

        Args:
            returns_dict (dict or np.ndarray): Dictionary of strategy names to returns arrays,
                                               or a single numpy array (treated as one strategy).
            initial_wealth (float, optional): Starting wealth for cumulative calculation. Defaults to 1.0.
            savefig (bool, optional): If True, saves the plot as a PNG file with a timestamped filename. Defaults to False.

        Behavior:
            - Converts a single numpy array input into a dictionary with key "Strategy".
            - Computes cumulative wealth as `initial_wealth * cumprod(1 + returns)`.
            - Plots each strategy's wealth trajectory on a logarithmic y-axis.
            - Adds a horizontal breakeven line at the initial wealth.
            - Displays the plot and optionally saves it to a PNG file.
        """
        if isinstance(returns_dict, np.ndarray):
            returns_dict = {"Strategy": returns_dict}
        plt.figure(figsize=(12, 6))
        for name, returns in returns_dict.items():
            wealth = initial_wealth * np.cumprod(1 + returns)
            plt.plot(wealth, label=name, linewidth=2)
        plt.yscale("log")
        plt.axhline(y=1, color='black', linestyle=':', label="Breakeven")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Wealth", fontsize=12)
        plt.title("Portfolio Wealth Over Time", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
        plt.tight_layout()
        if savefig:
            plt.savefig(f"plot_{int(time.time()*1000)}.png", dpi=300, bbox_inches='tight')
        plt.show()