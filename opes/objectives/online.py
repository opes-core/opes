"""
Online portfolio selection represents a fundamentally different paradigm from
traditional optimization: rather than assuming stationary distributions and
optimizing once, online algorithms sequentially update portfolio weights as new
data arrives, making no statistical assumptions about return distributions. This
framework emerged from machine learning and online learning theory, where the goal
is to compete with the best fixed strategy in hindsight while observing data
only once, in sequence.

!!! warning "Warning:"
    Certain online learning algorithms, currently `ExponentialGradient`,
    only uses the latest return data to update the weights. So, they might work
    suboptimally in backtests having `rebalance_freq` more than `1`.

---
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opes.objectives.base_optimizer import Optimizer
from ..regularizer import _find_regularizer
from ..utils import extract_trim, test_integrity, find_constraint
from ..errors import OptimizationError, PortfolioError

# Small epsilon value for numerical stability
EPSILON = 1e-8


class UniversalPortfolios(Optimizer):
    """
    Cover's Universal Portfolio.

    Cover's Universal Portfolio is a foundational algorithm in online portfolio
    selection that constructs a portfolio as a wealth-weighted average of all
    possible constant-rebalanced portfolios over the simplex. Introduced by
    Thomas Cover, it belongs to the family of universal portfolio algorithms,
    maintaining deterministically updated weights that asymptotically track the
    performance of the best constant-rebalanced portfolio in hindsight. The
    method leverages a combinatorial integration over the simplex or, in practice,
    approximates it via sampling or grid discretization, ensuring parameter-free
    portfolio growth in adversarial market environments.

    !!! warning "Warning"
        Grid-based Cover's UP scales combinatorially with the number of assets and
        `grid_res`. Large values may cause severe memory and runtime issues. Use with caution.

    """

    def __init__(self, technique="sample", n_samples=100, dirichlet_conc=1, grid_res=3):
        """
        **Args:**

        - `technique` (*str, optional*): Portfolio generation technique. Must be either `"sample"` or `"grid"`. Defaults to `"sample"`.
            - `"sample"`: Generates portfolios stochastically using a Dirichlet distribution.
            - `"grid"`: Generates portfolios deterministically using a simplex grid
        - `n_samples` (*int, optional*): Number of portfolios to sample when `technique="sample"`. Ignored when `technique="grid"`. Defaults to `100`.
        - `dirichlet_conc` (*float, optional*): Concentration parameter of the Dirichlet distribution used for stochastic portfolio sampling. Defaults to `1`.
        - `grid_res` (*int, optional*): Resolution parameter for simplex grid construction when `technique="grid"`. Smaller values are recommended due to combinatorial growth. Defaults to `3`.
        """
        self.identity = "cover-up"
        self.technique = technique.lower()
        self.n_samples = n_samples
        self.dirichlet_conc = dirichlet_conc
        self.grid_res = grid_res

        self.tickers = None
        self.weights = None

    def _return_sample_portfolios(self, rng_value):
        ticker_length = len(self.tickers)
        base_vector = np.full(ticker_length, self.dirichlet_conc)
        rng = np.random.default_rng(seed=rng_value)
        return rng.dirichlet(base_vector, size=self.n_samples)

    def _return_grid_portfolios(self):
        ticker_length = len(self.tickers)
        bars = np.array(
            list(
                combinations(
                    range(ticker_length + self.grid_res - 1), ticker_length - 1
                )
            )
        )
        counts = np.diff(
            np.c_[
                np.zeros(len(bars), int),
                bars + 1,
                np.full(len(bars), ticker_length + self.grid_res),
            ],
            axis=1,
        )
        return counts / self.grid_res

    def _prepare_inputs(self, data):

        # Quick validation check
        if self.technique not in ["sample", "grid"]:
            raise PortfolioError(
                f"Invalid technique. Expected 'sample' or 'grid', got {self.technique}"
            )

        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(
            np.ones(len(self.tickers)) / len(self.tickers), dtype=float
        )  # Placeholder, not important
        return data

    def optimize(self, data=None, seed=100, **kwargs):
        """
        Finds the Cover's Universal Portfolio:

        $$
        \\mathbf{w}_{\\text{cover}} = \\frac{ \\int_\\Delta \\mathbf{w} \\cdot W_T(\\mathbf{w}) \\ d\\mathbf{w}} {\\int_\\Delta W_T(\\mathbf{w}) \\ d\\mathbf{w}} \\approx \\frac{ \\sum_{\\mathbf{w} \\in \\mathcal P} \\mathbf{w} \\cdot W_T(\\mathbf{w})} {\\sum_{\\mathbf{w} \\in \\mathcal P} W_T(\\mathbf{w})}
        $$

        With $\\mathcal P \\sim \\text{Grid}(k)$ or $\\mathcal P \\sim \\text{Dirichlet}(\\alpha)$ depending upon user choice.

        !!! note "Note"
            Asset weight bounds are defaulted to (0,1).

        Args:
            data (*pd.DataFrame*): Ticker price data in either multi-index or single-index formats. Examples are given below:
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
                - Open
                - High
                - Low
                - Close
                - Volume
                ```
            seed (*int or None, optional*): Seed for dirichlet sampling. Defaults to `100` to preserve deterministic outputs.
            `**kwargs` (*optional*): Included for interface consistency, allowing the backtesting engine to pass additional or optimizer-specific arguments that may be safely ignored by this optimizer.

        **Returns:**

        - `np.ndarray`: Vector of optimized portfolio weights.

        Raises:
            DataError: For any data mismatch during integrity check.
            PortfolioError: For any invalid portfolio variable inputs during integrity check.

        !!! example "Example:"
            ```python
            # Importing the universal portfolios module
            from opes.objectives.online import UniversalPortfolios as UP

            # Let this be your ticker data
            training_data = some_data()

            # Initialize deterministic Cover's universal portfolio
            deterministic_up = UP(technique='grid', grid_res=5)

            # Initialize stochastic Cover's universal portfolio
            stochastic_up = UP(technique='sample', n_samples=1000, dirichlet_conc=1.5)

            # Optimize both portfolios
            det_up_weights = deterministic_up.optimize(data=training_data)
            stoch_up_weights = stochastic_up.optimize(data=training_data, seed=46)
            ```
        """
        # Preparing asset returns
        asset_returns = self._prepare_inputs(data)

        # Obtaining portfolios, either by sample or by grid
        # Note: 'else' is used since the attribute is checked whether its 'sample' or 'grid' in the _prepare_inputs() method
        if self.technique == "sample":
            portfolios = self._return_sample_portfolios(rng_value=seed)
        else:
            portfolios = self._return_grid_portfolios()

        # Computing portfolio returns, wealths and wealth weighted portfolio
        portfolio_returns = asset_returns @ portfolios.T
        portfolio_wealths = np.prod(1 + portfolio_returns, axis=0)
        wealth_weighted_portfolio = portfolio_wealths @ portfolios

        # Normalization and finding optimal weights
        self.weights = wealth_weighted_portfolio / portfolio_wealths.sum()
        return self.weights


class BCRP(Optimizer):
    """
    Best Constant Rebalanced Portfolio (BCRP).

    Introduced by Thomas Cover in his universal portfolio theory, the BCRP
    represents the optimal fixed-weight portfolio that rebalances to constant
    proportions after each period. BCRP is the gold standard benchmark in
    online portfolio selection: It achieves the maximum wealth that any
    constant-proportion strategy could have achieved over the observed sequence.
    """

    def __init__(self, reg=None, strength=1):
        """
        Args:
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.

        !!! tip "Tip:"
            Since both Follow-the-Leader (FTL) and Follow-the-Regularized-Leader (FTRL) compute the best constant rebalanced portfolio
            (BCRP) in hindsight to determine the allocation for the subsequent time step, both strategies can be implemented using the
            `BCRP` class.
        """
        self.identity = "bcrp"
        self.reg = _find_regularizer(reg)
        self.strength = strength

        self.tickers = None
        self.weights = None

    def _prepare_optimization_inputs(self, data, w):
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(
            np.ones(len(self.tickers)) / len(self.tickers) if w is None else w,
            dtype=float,
        )

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights)
        return data

    def optimize(self, data=None, w=None):
        """
        Solves the BCRP objective:

        $$
        \\min_{\\mathbf{w}} \\ - \\prod^T_t \\left(\\mathbf{w}^\\top \\mathbf{x}_t\\right) + \\lambda R(\\mathbf{w})
        $$

        Args:
            data (*pd.DataFrame*): Ticker price data in either multi-index or single-index formats. Examples are given below:
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
                - Open
                - High
                - Low
                - Close
                - Volume
                ```
            w (*None or np.ndarray, optional*): Initial weight vector for warm starts. Mainly used for backtesting and not recommended for user input. Defaults to `None`.

        **Returns:**

        - `np.ndarray`: Vector of optimized portfolio weights.

        Raises:
            DataError: For any data mismatch during integrity check.
            PortfolioError: For any invalid portfolio variable inputs during integrity check.
            OptimizationError: If `SLSQP` solver fails to solve.

        !!! example "Example:"
            ```python
            # Importing the BCRP module
            from opes.objectives.online import BCRP

            # Let this be your ticker data
            training_data = some_data()

            # Initialize BCRP for FTL and FTRL respectively
            ftl = BCRP()
            ftrl = BCRP(reg='entropy', strength=0.05)

            # Optimize both FTL and FTRL portfolios
            weights_ftl = ftl.optimize(data=training_data)
            weights_ftrl = ftrl.optimize(data=training_data)
            ```
        """
        # Preparing optimization and finding constraint
        # Bounds are defaulted to (0,1), constrained to the simplex
        trimmed_return_data = self._prepare_optimization_inputs(data, w)
        constraint = find_constraint(bounds=(0, 1))
        w = self.weights

        # Optimization objective and results
        def f(w):
            X = np.prod(1 + np.maximum(trimmed_return_data, -0.95) @ w)
            return -X + self.strength * self.reg(w)

        result = minimize(
            f, w, method="SLSQP", bounds=[(0, 1)] * len(w), constraints=constraint
        )
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(f"BCRP optimization failed: {result.message}")

    def set_regularizer(self, reg=None, strength=1):
        """
        Updates the regularization function and its penalty strength.

        This method updates both the regularization function and its associated
        penalty strength. It is primarily intended for strategies in which the
        regularization must change over time, such as in Follow-the-Regularized-
        Leader (FTRL) or other adaptive optimization procedures.

        Args:
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.

        !!! example "Example:"
            ```python
            # Import the BCRP class
            from opes.objectives.online import BCRP

            # Set with 'entropy' regularization
            ftrl = BCRP(reg='entropy', strength=0.01)

            # --- Do Something with `ftrl` ---
            ftrl.optimize(data=some_data())

            # Change regularizer using `set_regularizer`
            ftrl.set_regularizer(reg='l1', strength=0.02)

            # --- Do something else with new `ftrl` ---
            ftrl.optimize(data=some_data())
            ```
        """
        self.reg = _find_regularizer(reg)
        self.strength = strength


class ExponentialGradient(Optimizer):
    """
    Exponential Gradient (EG) optimizer.

    The Exponential Gradient algorithm is a foundational online learning algorithm
    that updates portfolio weights using multiplicative updates proportional to exponential returns.
    Introduced by Helmbold et. al, it belongs to the family of online convex optimization algorithms
    and maintains weights that rise exponentially with cumulative performance.
    """

    def __init__(self, learning_rate=0.3):
        """
        Args:
            learning_rate (*float, optional*): Learning rate for the EG algorithm. Usually bounded within (0,1]. Defaults to `0.3`.
        """
        self.identity = "expgrad"
        self.learning_rate = learning_rate

        self.tickers = None
        self.weights = None

    def _prepare_inputs(self, data, w):
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        # Defaulting previous weights to 1/N if none are given
        self.tickers, data = extract_trim(data)
        self.weights = np.array(
            np.ones(len(self.tickers)) / len(self.tickers) if w is None else w,
            dtype=float,
        )

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights)
        return data

    def optimize(self, data=None, w=None):
        """
        Performs the Exponential Gradient weight update rule.

        $$
        \\mathbf{w}_{i,t+1} = \\mathbf{w}_{i,t} \\cdot \\exp(\\eta \\cdot \\nabla f_{t,i})
        $$

        For this implementation, we have taken the reward function $f_{t} = \\log \\left(1 + \\mathbf{w}^\\top \\mathbf{r}_t\\right)$

        !!! note "Note"
            Asset weight bounds are defaulted to (0,1).

        Args:
            data (*pd.DataFrame*): Ticker price data in either multi-index or single-index formats. Examples are given below:
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
                - Open
                - High
                - Low
                - Close
                - Volume
                ```
            w (*None or np.ndarray, optional*): Previous weight vector for updation. If `None`, previous weights are assumed to be uniform weights. Defaults to `None`.

        **Returns:**

        - `np.ndarray`: Vector of optimized portfolio weights.

        Raises:
            DataError: For any data mismatch during integrity check.
            PortfolioError: For any invalid portfolio variable inputs during integrity check.

        !!! example "Example:"
            ```python
            # Importing the exponential gradient module
            from opes.objectives.online import ExponentialGradient as EG

            # Let this be your ticker data
            training_data = some_data()

            # Initialize exponential gradient with high learning rate
            e_g = EG(learning_rate=0.77)

            # Optimize for weights using a previous weight vector
            updated_weights = e_g.optimize(data=training_data, w=prev_weights())
            ```
        """
        # Preparing optimization and finding constraint
        # EG uses weight update method, so it takes the most recent (gross) return and uses it to update weights
        recent_return = self._prepare_inputs(data, w)[-1] + 1.0
        portfolio_return = recent_return @ self.weights

        # Capping to small epsilon value for numerical stability
        # Assets like GME (2021) can return huge negative values
        if portfolio_return < EPSILON:
            portfolio_return = EPSILON

        # Exponential Gradient update & normalization
        # We apply the log-sum-exp technique with subtracting the maximum to improve numerical stability
        # Weights are shift-invariant since they are exponentiated
        log_w = (
            np.log(self.weights + EPSILON)
            + self.learning_rate * recent_return / portfolio_return
        )
        log_w -= log_w.max()
        new_weights = np.exp(log_w)
        self.weights = new_weights / new_weights.sum()

        return self.weights
