"""
OPES contains a portfolio backtesting engine inclusive of a simple plotting function
using a supplied optimizer. The method supports both static and rolling portfolio
construction, applies transaction costs and enforces strict no-lookahead
constraints during rebalancing.

The backtest operates exclusively on the test dataset while ensuring that
all portfolio decisions are made using only information available at the time of execution.

---
"""

from numbers import Real, Integral as Integer
import time
import inspect

import numpy as np
import pandas as pd
import scipy.stats as scistats
import matplotlib.pyplot as plt

from opes.errors import PortfolioError, DataError
from opes.utils import extract_trim


class Backtester:
    """
    A comprehensive backtesting engine for financial time series.

    This class manages training and testing datasets, ensuring that
    any missing values (NaNs) are removed for robust backtesting loops.
    It also stores transaction cost parameters for portfolio simulations.
    """

    def __init__(self, train_data=None, test_data=None, cost={"const": 10.0}):
        """
        **Args:**

        - `train_data` (*pd.DataFrame*): Historical training data. Defaults to `None`.
        - `test_data` (*pd.DataFrame*): Historical testing data. Defaults to `None`.
        - `cost` (*dict, optional*): Transaction cost parameters. Defaults to `{'const': 10.0}`. Various cost models are given below:
            - `{'const': constant_bps_value}`: Constant cost value throughout time. Deterministic.
            - `{'gamma': (shape, scale)}`: Gamma distributed cost. Stochastic.
            - `{'lognormal': (mu, sigma)}`: Lognormally distributed cost. Stochastic.
            - `{'inversegaussian': (mean, shape)}`: Inverse gaussian distributed cost. Stochastic.
            - `{'jump': (arrival_rate, mu, sigma)}`: Poisson-compound lognormally distributed cost. Stochastic.

        !!! note "Notes:"
            - `train_data` and `test_data` must be of the same format: either Single-Index with `DateTimeIndex` and tickers as columns or Multi-Index with a necessary "Close" column in level 1 and tickers in level 0. Examples are shown below
            ```
            # Single-Index Example
            Ticker           TSLA      NVDA       GME        PFE       AAPL  ...
            Date
            2015-01-02  14.620667  0.483011  6.288958  18.688917  24.237551  ...
            2015-01-05  14.006000  0.474853  6.460137  18.587513  23.554741  ...
            2015-01-06  14.085333  0.460456  6.268492  18.742599  23.556952  ...
            2015-01-07  14.063333  0.459257  6.195926  18.999102  23.887287  ...
            2015-01-08  14.041333  0.476533  6.268492  19.386841  24.805082  ...
            ...

            # Multi-Index Example Structure (OHLCV)
            Columns:
            + Ticker (e.g. GME, PFE, AAPL, ...)
              + Open
              + High
              + Low
              + Close
              + Volume
            ```
            - NaN values in both `train_data` and `test_data` are automatically dropped to prevent indexing errors during backtests. So Assets with shorter trading histories truncate the dataset.
            - After cleaning/truncation, close prices are extracted per asset. Returns are computed as:

            $$R_t = \\frac{P^{(t)}}{P^{(t-1)}} - 1$$
        """
        # Assigning by dropping nans to ensure proper indexing
        # Dropping nan rows results makes backtest loops robust and predictable
        self.train = train_data.dropna()
        self.test = test_data.dropna()
        self.cost = cost

    def _backtest_integrity_check(
        self, optimizer, rebalance_freq, reopt_freq, seed, cleanweights=False
    ):
        # Checking train and test data validity
        if not isinstance(self.train, pd.DataFrame):
            raise DataError(
                f"Invalid train data type. Expected pd.DataFrame, got {type(self.train)}"
            )
        if not isinstance(self.test, pd.DataFrame):
            raise DataError(
                f"Invalid test data type. Expected pd.DataFrame, got {type(self.train)}"
            )
        # Checking train and test data length and format
        if len(self.train) < 5:
            raise DataError(
                f"Insufficient training data for backtest. Expected len(data) >= 5, got {len(self.train)}"
            )
        if len(self.train) <= 0:
            raise DataError(
                f"Insufficient training data for backtest. Expected len(data) > 0, got {len(self.train)}"
            )
        if not ((self.train.columns.equals(self.test.columns))):
            raise DataError("Train and test DataFrames have different formats")
        if cleanweights is not True and cleanweights is not False:
            raise DataError(
                f"Invalid cleanweights variable. Expected True or False, got {cleanweights}"
            )
        # Checking optimizer validity
        if not hasattr(optimizer, "optimize"):
            raise PortfolioError(
                f"Expected optimizer object to have 'optimize' attribute."
            )
        # Checking rebalance frequency type and validity
        if not isinstance(rebalance_freq, Integer) or rebalance_freq <= 0:
            raise PortfolioError(
                f"Invalid rebalance frequency. Expected integer within bounds [1,T], Got {rebalance_freq}"
            )
        # Checking re-optimization frequency type validity
        if not isinstance(reopt_freq, Integer) or reopt_freq <= 0:
            raise PortfolioError(
                f"Invalid re-optimization frequency. Expected integer within bounds [1,T], Got {reopt_freq}"
            )
        # Validiating numpy seed
        if seed is not None and not isinstance(seed, int):
            raise PortfolioError(f"Invalid seed. Expected integer or None, Got {seed}")
        # Cost model validity - per model check
        if len(self.cost) != 1:
            raise PortfolioError(
                f"Invalid cost model. Cost model must be a dictionary of length 1, Got {len(self.cost)}"
            )
        first_key = next(iter(self.cost))
        first_key_low = first_key.lower()
        if first_key_low not in [
            "const",
            "lognormal",
            "gamma",
            "inversegaussian",
            "jump",
        ]:
            raise PortfolioError(f"Unknown cost model: {first_key}")
        elif first_key_low == "const" and not isinstance(self.cost[first_key], Real):
            raise PortfolioError(
                f"Unspecified cost value. Expected real number, got {type(self.cost[first_key])}"
            )
        elif (
            first_key_low in ["lognormal", "gamma", "inversegaussian"]
            and len(self.cost[first_key]) != 2
        ):
            raise PortfolioError(
                f"Invalid cost model parameter length. Expected 2, got {len(self.cost[first_key])}"
            )
        elif first_key_low == "jump" and len(self.cost[first_key]) != 3:
            raise PortfolioError(
                f"Invalid jump cost model parameter length. Expected 3, got {len(self.cost[first_key])}"
            )

    # Helper method to combine datasets
    # Combines training and testing data upto a particular timestep
    def _combine_datasets(self, upto_timestep):
        # NO LOOKAHEAD BIAS
        # Rebalance at timestep t using only past data (up to t, exclusive) to avoid lookahead bias
        # Training data is pre-cleaned (no NaNs), test data up to t is also NaN-free
        # Concatenating them preserves this property; dropna() handles edge cases safely
        # The optimizer therefore only sees information available until the current decision point
        combined_dataset = pd.concat([self.train, self.test.iloc[:upto_timestep]])
        combined_dataset = combined_dataset[
            ~combined_dataset.index.duplicated(keep="first")
        ].dropna()

        return combined_dataset

    # Helper method to compute drifted weights for timestep
    # Returns the realized drifted weights for the timestep
    def _compute_drifted_weights(self, w_prev, returns):
        w_prev = np.asarray(w_prev)
        w_realized = (w_prev * (1 + returns)) / (1 + np.sum(w_prev * returns))

        return w_realized

    # Helper method to compute costs array
    # Returns either a constant or an array of length=horizon of cost
    def _slippage_costs(self, cost, horizon, numpy_seed=None):
        # Setting numpy seed and finding cost parameters
        numpy_rng = np.random.default_rng(numpy_seed)
        cost_key = next(iter(cost)).lower()
        cost_params = cost[cost_key]

        # ---------- COST MODELS ----------
        match cost_key:
            # Constant slippage
            case "const":
                return cost_params / 10000
            # Gamma distributed slippage
            case "gamma":
                return (
                    numpy_rng.gamma(
                        shape=cost_params[0], scale=cost_params[1], size=horizon
                    )
                    / 10000
                )
            # Lognormally distributed slippage
            case "lognormal":
                return (
                    numpy_rng.lognormal(
                        mean=cost_params[0], sigma=cost_params[1], size=horizon
                    )
                    / 10000
                )
            # Inverse gaussian slippage
            case "inversegaussian":
                return (
                    numpy_rng.wald(
                        mean=cost_params[0], scale=cost_params[1], size=horizon
                    )
                    / 10000
                )
            # Compound-poisson lognormal slippage (jump process)
            case "jump":
                N = numpy_rng.poisson(cost_params[0], size=horizon)
                jump_cost = np.array(
                    [
                        (
                            np.sum(
                                numpy_rng.lognormal(
                                    mean=cost_params[1], sigma=cost_params[2], size=n
                                )
                            )
                            if n > 0
                            else 0
                        )
                        for n in N
                    ]
                )
                return jump_cost / 10000
            case _:
                raise DataError(f"Unknown cost model: {cost_key}")

    def backtest(
        self,
        optimizer,
        rebalance_freq=1,
        reopt_freq=1,
        seed=100,
        weight_bounds=None,
        clean_weights=False,
    ):
        """
        Execute a portfolio backtest over the test dataset using a given optimizer.

        This method performs a walk-forward backtest using the user defined `rebalance_freq`
        and `reopt_freq`. It also applies transaction costs and ensures no lookahead bias.
        For a rolling backtest, any common date values are dropped, the first occurrence
        is considered to be original and kept.

        !!! warning "Warning:"
            Some online learning methods such as `ExponentialGradient` update weights based
            on the most recent observations. Setting `reopt_freq` to any value other
            than `1` may result in suboptimal performance, as intermediate data points will
            be ignored and not used for weight updates.

        **Args:**

        - `optimizer`: An optimizer object containing the optimization strategy. Accepts both OPES built-in objectives and externally constructed optimizer objects.
        - `rebalance_freq` (*int, optional*): Frequency of rebalancing in time steps. Must be `>= 1`. Defaults to `1`.
        - `reopt_freq` (*int, optional*): Frequency of re-optimization in time steps. Must be `>= 1`. Defaults to `1`.
        - `seed` (*int or None, optional*): Random seed for reproducible cost simulations. Defaults to `100`.
        - `weight_bounds` (*tuple, optional*): Bounds for portfolio weights passed to the optimizer if supported.

        !!! abstract "Rules for `optimizer` Object"
            - `optimizer` Must contain `optimize(data, **kwargs)` attribute which is functional.
            - `optimize(data, **kwargs)` method must contain the following parameters:
                - `data`: OHLCV, multi-index or single-index pandas DataFrame.
                - `**kwargs`: For safety against breaking changes.
            - `optimize` must output weights for the timestep.

        !!! note "Note"
            - The backtest assumes portfolio weights are applied at the open of each timestep, with zero execution delay.
            - Re-optimization does not automatically imply rebalancing. When the portfolio is re-optimized at a given timestep, weights may or may not be updated depending on the value of `rebalance_freq`.
            - To ensure a coherent backtest, a common practice is to choose frequencies such that `reopt_freq % rebalance_freq == 0`. This guarantees that whenever optimization occurs, a rebalance is also performed.
            - Also note that within a given timestep, rebalancing, if it occurs, is performed after optimization when optimization is scheduled for that timestep.

        !!! tip "Tip"
            Common portfolio styles can be constructed by appropriate choices of `rebalance_freq` and `reopt_freq`:

            - Buy-and-Hold: `rebalance_freq > horizon`, `reopt_freq > horizon`
            - Constantly Rebalanced: `rebalance_freq = 1`, `reopt_freq > horizon`
            - Fully Dynamic: `rebalance_freq = 1`, `reopt_freq = 1`

        **Returns:**

        - `dict`: Backtest results containing the following keys:
            - `'returns'` (*np.ndarray*): Portfolio returns after accounting for costs.
            - `'weights'` (*np.ndarray*): Portfolio weights at each timestep.
            - `'costs'` (*np.ndarray*): Transaction costs applied at each timestep.
            - `'timeline'` (*np.ndarray*): Timeline on which the backtest was conducted.

        Raises:
            DataError: If the optimizer does not accept weight bounds but `weight_bounds` are provided.
            PortfolioError: If input validation fails (via `_backtest_integrity_check`).
            OptimizationError: If the underlying optimizer uses optimization and if it fails to optimize.

        !!! note "Notes:"
            - All returned arrays are aligned in time and have length equal to the test dataset.
            - Returns and weights are stored in arrays aligned with test data indices.

        !!! example "Example:"
            ```python
            import numpy as np

            # Importing necessary OPES modules
            from opes.objectives import Kelly
            from opes import Backtester

            # Place holder for your price data
            from some_random_module import trainData, testData

            # Declare train and test data
            training = trainData()
            testing = testData()

            # Declaring kelly
            kelly_optimizer = Kelly(fraction=0.8)

            # Initializing Backtest with constant costs
            tester = Backtester(train_data=training, test_data=testing)

            # Obtaining backtest data for kelly optimizer
            kelly_backtest = tester.backtest(
                optimizer=kelly_optimizer,
                rebalance_freq=1,  # Rebalance daily
                reopt_freq=21      # Re-optimize monthly
            )

            # Printing results
            for key in kelly_backtest:
                print(f"{key}: {kelly_backtest[key]}")
            ```
        """
        # Running backtester integrity checks, extracting test return data and caching values
        self._backtest_integrity_check(
            optimizer, rebalance_freq, reopt_freq, seed, cleanweights=clean_weights
        )
        test_data = extract_trim(self.test)[1]
        optimizer_parameters = inspect.signature(optimizer.optimize).parameters
        horizon = len(test_data)

        # ---------- BACKTEST LOOP ----------

        # Initializing weights list and turnover array
        # NOTE: More readable than initializing a 2D numpy array
        weights = [None] * horizon
        turnover_array = np.zeros(horizon)

        # First optimization is done manually using training data
        # Using weight bounds if it is given AND if it is present as a parameter within optimize method
        # Otherwise weights are optimized without weight bounds argument
        kwargs = {}
        # Checking for weight_bounds
        if weight_bounds is not None and "weight_bounds" in optimizer_parameters:
            kwargs["weight_bounds"] = weight_bounds
        # Optimizing for the timestep
        optimized_weights = optimizer.optimize(self.train, **kwargs)

        # Cleaning weights if true and if optimizer has method
        if clean_weights and hasattr(optimizer, "clean_weights"):
            optimized_weights = optimizer.clean_weights()

        # Assigning computed weights to weight array
        weights[0] = optimized_weights
        optimizer_parameters = optimizer_parameters

        # For loop through timesteps to automate remaining walk-forward test
        for t in range(1, horizon):

            # ---------- RE-OPTIMIZATION BLOCK ----------
            # Re-optimization check during appropriate frequency
            # If the check is satisfied optimization is taken place and the new weights are computed
            # NOTE: Rebalancing is handled separately using `rebalance_freq`
            if t % reopt_freq == 0:

                combined_dataset = self._combine_datasets(upto_timestep=t)

                # We find if 'w' and 'weight_bounds' parameters are present within the optimizer
                # The parameters which are present are leveraged (Eg. warm start, weight updates for 'w')
                # Otherwise it is optimized without any extra arguments
                kwargs = {}
                if "w" in optimizer_parameters:
                    kwargs["w"] = optimized_weights
                if (
                    weight_bounds is not None
                    and "weight_bounds" in optimizer_parameters
                ):
                    kwargs["weight_bounds"] = weight_bounds

                # Optimizing for the timestep
                optimized_weights = optimizer.optimize(combined_dataset, **kwargs)

                # Cleaning weights if true and if optimizer has method
                if clean_weights and hasattr(optimizer, "clean_weights"):
                    optimized_weights = optimizer.clean_weights()

            # ---------- REBALANCING BLOCK ----------
            # Computing drifted weights
            # This is necessary for turnover and slippage modelling
            # NOTE: weights and returns of the previous timestep are passed in to compute drifted weights
            # This is because weights[0], which is to be set on the beginning of the  zeroth day is separately computed
            # Therefore, as the loop starts from 1, the return from the zeroth day will cause the first drifted weights on the start of the first day (end of zeroth day)
            drifted_weights = self._compute_drifted_weights(
                weights[t - 1], test_data[t - 1]
            )

            # Assigning computed weights to weight array
            # If rebalance frequency is satisfied, then the weights for the timestep is the optimized weights
            # Otherwise, the weights for the timestep is the drifted (realized) weights
            if t % rebalance_freq == 0:
                weights[t] = optimized_weights
            else:
                weights[t] = drifted_weights

            # ---------- TURNOVER BLOCK ----------
            # Computing turnover
            # turnover is the L1 distance from current weights to drifted weights
            # If not rebalanced, turnover is 0
            turnover_for_timestep = np.sum(np.abs(weights[t] - drifted_weights))
            turnover_array[t] = turnover_for_timestep

        # --------- POST PROCESSING BLOCK ---------
        # Creating vertical stack for vectorization
        weights_array = np.vstack(weights)
        # Computing slippage costs over time, vectorizing portfolio returns and finding final portfolio returns after costs
        costs_array = turnover_array * self._slippage_costs(
            cost=self.cost, horizon=horizon, numpy_seed=seed
        )
        portfolio_returns = np.einsum("ij,ij->i", weights_array, test_data)
        portfolio_returns -= costs_array
        # Finding timeline array from test data
        # NOTE: the first value is excluded since pct_change() drops the first date for return construction
        timeline_array = self.test.index.to_numpy()[1:]

        return {
            "returns": portfolio_returns,
            "weights": weights_array,
            "costs": costs_array,
            "timeline": timeline_array,
        }

    def get_metrics(self, returns):
        """
        Computes a comprehensive set of portfolio performance metrics from returns.

        This method calculates risk-adjusted and absolute performance measures
        commonly used in finance, including volatility, drawdowns and tail risk metrics.

        Args:
            returns (*array-like*): Array or list of periodic portfolio returns. Will be converted to numpy array.

        **Returns:**

        - `dict`: Dictionary of performance metrics with the following keys:
            - `'sharpe'`: Sharpe ratio.
            - `'sortino'`: Sortino ratio.
            - `'volatility'`: Standard deviation of returns (%).
            - `'growth_rate'` : Geometric mean growth of the portfolio (%).
            - `'mean_return'`: Mean return (%).
            - `'total_return'`: Total cumulative return (%).
            - `'mean_drawdown'` : Mean drawdown (%).
            - `'max_drawdown'`: Maximum drawdown (%).
            - `'ulcer_index'` : Ulcer index.
            - `'var_95'`: Value at Risk at 95% confidence level (%).
            - `'cvar_95'`: Conditional Value at Risk (expected shortfall) at 95% (%).
            - `'skew'`: Skewness of returns.
            - `'kurtosis'`: Kurtosis of returns.
            - `'omega_0'`: Omega ratio (gain/loss ratio).
            - `'hit_ratio'` : Hit ratio.

        !!! note "Notes"
            - The following metrics are scaled to percentages:
                - `'volatility'`
                - `'growth_rate'`
                - `'mean_return'`
                - `'total_return'`
                - `'mean_drawdown'`
                - `'max_drawdown'`
                - `'var_95'`
                - `'cvar_95'`

            - The following metrics are returned as loss values (usually positive):
                - `'mean_drawdown'`
                - `'max_drawdown'`
                - `'ulcer_index'`
                - `'var_95'`
                - `'cvar_95'`

            - All metrics are rounded to 5 decimal places.

        !!! example "Example:"
            ```python
            # Importing portfolio method and backtester
            from opes.objectives import MaxSharpe
            from opes import Backtester

            # Place holder for your price data
            from some_random_module import trainData, testData

            # Declare train and test data
            training = trainData()
            testing = testData()

            # Declare the maximum sharpe optimizer with risk-free rate of 0.02
            max_sharpe_opt = MaxSharpe(risk_free=0.02)

            # Initializing Backtest with constant costs
            tester = Backtester(train_data=training, test_data=testing)

            # Obtaining returns array from backtest and finding metrics
            optimizer_returns = tester.backtest(optimizer=max_sharpe_opt)['returns']
            metrics = tester.get_metrics(optimizer_returns)

            # Printing sharpe and maximum drawdown
            print(metrics["sharpe"], metrics["max_drawdown"])
            ```
        """
        # Converting returns to numpy array and cleaning
        returns = np.array(returns)
        returns = returns[np.isfinite(returns)]

        # Caching repeated values
        downside_vol = returns[returns < 0].std()
        vol = returns.std()
        drawdowns = 1 - np.cumprod(1 + returns) / np.maximum.accumulate(
            np.cumprod(1 + returns)
        )
        mean_ret = returns.mean()
        var = -np.quantile(returns, 0.05)
        tail_returns = returns[returns <= -var]

        # Performance metrics
        performance_metrics = {
            "sharpe": (mean_ret / vol if (vol > 0 and np.isfinite(vol)) else np.nan),
            "sortino": (
                mean_ret / downside_vol
                if (downside_vol > 0 and np.isfinite(downside_vol))
                else np.nan
            ),
            "volatility": vol * 100 if (vol > 0 and np.isfinite(vol)) else np.nan,
            "growth_rate": (np.prod(1 + returns) ** (1 / len(returns)) - 1) * 100,
            "mean_return": mean_ret * 100,
            "total_return": (np.prod(1 + returns) - 1) * 100,
            "max_drawdown": (np.max(drawdowns)) * 100,
            "mean_drawdown": (np.mean(drawdowns)) * 100,
            "ulcer_index": np.sqrt(np.mean(drawdowns**2)),
            "var_95": var * 100,
            "cvar_95": -tail_returns.mean() * 100 if len(tail_returns) > 0 else np.nan,
            "skew": scistats.skew(returns),
            "kurtosis": scistats.kurtosis(returns),
            "omega_0": np.sum(np.maximum(returns, 0)) / np.sum(np.maximum(-returns, 0)),
            "hit_ratio": np.mean(returns > 0),
        }

        # Rounding values to 5 decimal places
        for key in performance_metrics:
            performance_metrics[key] = round(performance_metrics[key], 5)

        return performance_metrics

    def plot_wealth(
        self, returns_dict, timeline=None, initial_wealth=1.0, savefig=False
    ):
        """
        OPES ships with a basic plotting utility for visualizing portfolio wealth over time.

        This method exists for quick inspection and debugging, not for deep performance analysis.
        It visualizes cumulative wealth for one or multiple strategies using their periodic
        returns. It also provides a breakeven reference line and optional saving of the plot to
        a file.

        !!! tip "Recommendation:"
            For serious research, reporting, or strategy comparison, we strongly recommend writing your own custom plotting pipeline.
            Real evaluation usually needs rolling Sharpe, drawdowns, volatility regimes, benchmark overlays and other diagnostics that
            go far beyond a single equity curve.

        Args:
            returns_dict (*dict or np.ndarray*): Dictionary of strategy names to returns arrays or a single numpy array (treated as one strategy).
            timeline (*None or array-like*): Sequence of dates corresponding to the portfolio backtest timeline. If `None`, numbers are used for the x-axis. Defaults to `None`.
            initial_wealth (*float, optional*): Starting wealth for cumulative calculation. Defaults to `1.0`.
            savefig (*bool, optional*): If `True`, saves the plot as a PNG file with a timestamped filename. Defaults to `False`.

        !!! note "Notes:"
            - Ensure `timeline` and `returns_dict[key]` lengths match.
            - Converts a single numpy array input into a dictionary with key "Strategy".
            - Computes cumulative wealth as $W_t = W_0 \\prod_{i}^T(1+r_i)$.
            - Plots each strategy's wealth trajectory on a logarithmic y-axis.
            - Adds a horizontal breakeven line at the initial wealth.
            - Displays the plot and optionally saves it to a PNG file.

        !!! example "Example:"
            ```python
            # Importing portfolio methods and backtester
            from opes.objectives import MaxMean, MeanVariance
            from opes import Backtester

            # Place holder for your price data
            from some_random_module import trainData, testData

            # Declare train and test data
            training = trainData()
            testing = testData()

            # Declare two optimizers
            maxmeanl2 = MaxMean(reg="l2", strength=0.001)
            mvo1_5 = MeanVariance(risk_aversion=1.5)

            # Initializing Backtest with constant costs
            tester = Backtester(train_data=training, test_data=testing)

            # Obtaining returns array from backtest for both optimizers
            scenario_1 = tester.backtest(optimizer=maxmeanl2)
            scenario_2 = tester.backtest(optimizer=mvo1_5)['returns']

            # Plotting wealth
            tester.plot_wealth(
                {
                    "Maximum Mean (L2, 1e-3)": scenario_1['returns'],
                    "Mean Variance (RA=1.5)": scenario_2,
                },
                timeline=scenario_1['timeline']
            )
            ```
        """
        if isinstance(returns_dict, np.ndarray):
            returns_dict = {"Strategy": returns_dict}
        plt.figure(figsize=(12, 6))
        for name, returns in returns_dict.items():
            wealth = initial_wealth * np.cumprod(1 + returns)
            plt.plot(timeline, wealth, label=name, linewidth=2)
        plt.yscale("log")
        plt.axhline(y=1, color="black", linestyle=":", label="Breakeven")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Wealth", fontsize=12)
        plt.title("Portfolio Wealth Over Time", fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        plt.tight_layout()
        if savefig:
            plt.savefig(
                f"plot_{int(time.time()*1000)}.png", dpi=300, bbox_inches="tight"
            )
        plt.show()
