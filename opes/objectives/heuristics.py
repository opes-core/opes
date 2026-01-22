"""
While utility theory and Markowitz optimization boast mathematical elegance,
the real-world of quantitative investing often prefers simpler heuristics because
estimating returns and covariances accurately from noisy data is basically guessing with style.
Heuristic portfolios dodge this problem by embracing simplicity, with some relying on trader
intuition like equal-weighting assets when *"who knows what will win"* and others using principled
rules such as spreading risk evenly or minimizing portfolio entropy. The common thread is
robustness, as avoiding strong assumptions about return predictions often leads these approaches
to outperform their theoretically optimal cousins outside the textbook.

---
"""

from numbers import Integral as Integer

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from opes.objectives.base_optimizer import Optimizer
from ..utils import extract_trim, test_integrity, find_constraint
from ..regularizer import _find_regularizer
from ..errors import OptimizationError, PortfolioError, DataError


class Uniform(Optimizer):
    """
    Uniform weight portfolio.

    The Uniform or 1/N portfolio is the simplest diversification strategy: allocate capital
    equally across all available assets. Introduced implicitly throughout financial history
    and formalized in academic comparisons by DeMiguel et. al., this approach makes no
    assumptions about relative asset qualities. Despite its simplicity, or perhaps because
    of it, equal weighting has proven surprisingly difficult to beat out-of-sample. The
    strategy completely avoids estimation error in expected returns and correlations,
    trading off theoretical optimality for robust performance.
    """

    def __init__(self):
        """
        The `Uniform` optimizer does not require any parameters to initialize.
        """
        self.identity = "uniform"

        self.tickers = None
        self.weights = None

    def _extract_tickers(self, data, n_assets):
        if data is None and n_assets is None:
            raise DataError("Portfolio data not specified.")

        elif data is not None and isinstance(data, list):
            self.tickers = data
        elif data is not None and isinstance(data, pd.DataFrame):
            self.tickers = data.columns.get_level_values(0).unique().tolist()
        elif n_assets is not None and isinstance(n_assets, Integer):
            self.tickers = ["UNKNOWN"] * n_assets
        else:
            raise DataError("Unsupported data format for tickers.")

    def optimize(self, data=None, n_assets=None, **kwargs):
        """
        Satisfies the 1/N objective:

        $$
        \\mathbf{w}_i = \\frac{1}{N} \\; \\forall \\ i=1, ..., N
        $$

        !!! note "Note"
            Asset weight bounds are defaulted to (0,1).

        Args:
            data (*list, pd.DataFrame or None, optional*): List of tickers or ticker price data in either multi-index or single-index formats. Examples are given below:
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
            n_assets (*int or None, optional*): Number of assets in the portfolio. If this is provided while `data` is `None`, a placeholder ticker list such as `["UNKNOWN", "UNKNOWN", ...]` is automatically generated. Defaults to `None`.
            `**kwargs` (*optional*): Included for interface consistency, allowing the backtesting engine to pass additional or optimizer-specific arguments that may be safely ignored by this optimizer.

        **Returns:**

        - `np.ndarray`: Vector of equal portfolio weights.

        Raises:
            DataError: For any data mismatch during integrity check.

        !!! example "Example:"
            ```python
            # Importing the equal-weight module
            from opes.objectives import Uniform

            # Let this be your ticker data
            training_data = some_data()

            # Initialize
            equal_weights = Uniform()

            # Optimize portfolio with ticker data
            weights = equal_weights.optimize(data=training_data)

            # Alternatively, optimize for fixed number of assets, here 5
            another_weights = equal_weights.optimize(n_assets=5)
            ```
        """
        self._extract_tickers(data, n_assets)

        # Assigning weights and returning the same
        self.weights = np.array(np.ones(len(self.tickers)) / len(self.tickers))
        return self.weights.copy()


class InverseVolatility(Optimizer):
    """
    Inverse Volatility portfolio.

    The inverse volatility portfolio, a practical simplification of minimum
    variance strategies used by practitioners since at least the 1970s,
    weights assets inversely proportional to their volatilities. The approach
    is grounded in the intuition that higher-risk assets should receive
    smaller allocations, which is equivalent to risk parity when all assets
    are uncorrelated. While deliberately naive about correlations, inverse
    volatility portfolios are trivial to compute, require minimal data and
    often perform surprisingly well out-of-sample.
    """

    def __init__(self):
        """
        The `InverseVolatility` optimizer does not require any parameters to initialize.
        """
        self.identity = "invvol"
        self.volarray = None

        self.tickers = None
        self.weights = None

    def _prepare_inputs(self, data):
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)

        # Extracting volatility array and testing integrity
        self.volarray = np.std(data, axis=0, ddof=1)
        test_integrity(tickers=self.tickers, volatility_array=self.volarray)

    def optimize(self, data=None, **kwargs):
        """
        Satisfies the Inverse Volatility objective:

        $$
        \\mathbf{w}_i = \\frac{1/\\sigma_i}{\\sum_j^N 1/\\sigma_j}
        $$

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
            `**kwargs` (*optional*): Included for interface consistency, allowing the backtesting engine to pass additional or optimizer-specific arguments that may be safely ignored by this optimizer.

        **Returns:**

        - `np.ndarray`: Vector of optimized portfolio weights.

        Raises:
            DataError: For any data mismatch during integrity check.

        !!! example "Example:"
            ```python
            # Importing the Inverse Volatility Portfolio (IVP) module
            from opes.objectives import InverseVolatility as IVP

            # Let this be your ticker data
            training_data = some_data()

            # Initialize portfolio
            inverse_vol = IVP()

            # Optimize portfolio
            weights = inverse_vol.optimize(data=training_data)
            ```
        """
        # Preparing inputs for finding weights
        self._prepare_inputs(data)
        self.weights = (1 / self.volarray) / (1 / self.volarray).sum()
        return self.weights.copy()


class SoftmaxMean(Optimizer):
    """
    Softmax Mean portfolio.

    The softmax mean portfolio, introduced in recent machine learning-inspired approaches
    to portfolio construction, applies the softmax function to expected returns to determine
    weights. This method exponentially scales weights based on expected returns through a
    temperature parameter $\\tau$, providing a smooth interpolation between equal weighting
    ($\\tau \\to \\infty$) and maximum mean return ($\\tau \\to 0$). This approach borrows
    from the exploration-exploitation tradeoff in reinforcement learning, offering a
    principled way to balance return-seeking with diversification.
    """

    def __init__(self, temperature=1):
        """
        Args:
            temperature (*float, optional*): Scalar that controls the sensitivity of the weights to return differences. Must be greater than `0`. Defaults to `1.0`.
        """
        self.identity = "softmean"
        self.mean = None
        self.temperature = temperature

        self.tickers = None
        self.weights = None

    def _prepare_inputs(self, data, custom_mean=None):
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)

        # Quick check for temperature validity
        if self.temperature <= 0:
            raise PortfolioError(
                f"Invalid temperature. Expected within bounds (0, inf), got {self.temperature})"
            )

        # Extracting mean and testing integrity
        self.mean = np.mean(data, axis=0) if custom_mean is None else custom_mean
        test_integrity(tickers=self.tickers, mean=self.mean)

    def optimize(self, data=None, custom_mean=None, **kwargs):
        """
        Satisfies the Softmax Mean objective:

        $$
        \\mathbf{w}_i = \\frac{\\exp\\left( \\mu_i / \\tau \\right)}{\\sum_j^N \\exp\\left( \\mu_i / \\tau \\right)}
        $$

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
            custom_mean (*None or np.ndarray, optional*): Custom mean vector. Can be used to inject externally generated mean vectors (eg. Black-Litterman). Defaults to `None`.
            `**kwargs` (*optional*): Included for interface consistency, allowing the backtesting engine to pass additional or optimizer-specific arguments that may be safely ignored by this optimizer.

        **Returns:**

        - `np.ndarray`: Vector of optimized portfolio weights.

        Raises:
            DataError: For any data mismatch during integrity check.

        !!! example "Example:"
            ```python
            # Importing the softmax mean module
            from opes.objectives import SoftmaxMean

            # Let this be your ticker data
            training_data = some_data()

            # Let this be your custom mean vector
            # Can be Black-Litterman, Bayesian, Fama-French etc.
            mean_v = customMean()

            # Initialize softmax mean
            soft = SoftmaxMean()

            # Optimize portfolio with custom mean vector
            weights = soft.optimize(data=training_data, custom_mean=mean_v)
            ```
        """
        # Preparing inputs
        self._prepare_inputs(data, custom_mean=custom_mean)

        # Solving weights
        self.weights = np.exp(
            self.mean / self.temperature - np.max(self.mean / self.temperature)
        )
        self.weights /= self.weights.sum()
        return self.weights.copy()


class MaxDiversification(Optimizer):
    """
    Maximum Diversification optimization.

    The maximum diversification portfolio, introduced by Yves Choueifaty and Yves Coignard,
    maximizes the diversification ratio, defined as the weighted average of asset volatilities
    divided by portfolio volatility. Motivated by the low-volatility anomaly, the strategy
    explicitly targets portfolios where diversification benefits are strongest, meaning
    portfolio volatility is significantly lower than the average individual volatility. It
    relies only on the covariance matrix and naturally yields well-diversified portfolios
    without extreme concentration, offering a practical alternative to minimum variance while
    preserving broad asset exposure.
    """

    def __init__(self, reg=None, strength=1):
        """
        Args:
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "maxdiverse"
        self.reg = _find_regularizer(reg)
        self.strength = strength
        self.covariance = None
        self.volarray = None

        self.tickers = None
        self.weights = None

    def _prepare_optimization_inputs(self, data, custom_cov=None):
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)

        # Checking for covariance, per-asset volatility and weights
        if custom_cov is None:
            # Handling invertibility using the small epsilon * identity matrix
            # small epsilon scales with the trace of the covariance
            self.covariance = np.cov(data, rowvar=False)
            epsilon = 1e-3 * np.trace(self.covariance) / self.covariance.shape[0]
            self.covariance = self.covariance + epsilon * np.eye(
                self.covariance.shape[0]
            )
        else:
            self.covariance = custom_cov
        self.volarray = np.sqrt(np.diag(self.covariance))
        self.weights = np.array(
            np.ones(len(self.tickers)) / len(self.tickers), dtype=float
        )

        # Functions to test data integrity
        test_integrity(
            tickers=self.tickers,
            weights=self.weights,
            cov=self.covariance,
            volatility_array=self.volarray,
        )

    def optimize(self, data=None, custom_cov=None, seed=100, **kwargs):
        """
        Solves the Maximum Diversification objective:

        $$
        \\min_{\\mathbf{w}} \\ -\\frac{\\mathbf{w}^\\top \\sigma}{\\sqrt{\\mathbf{w}^\\top \\Sigma \\mathbf{w}}} + \\lambda R(\\mathbf{w})
        $$

        !!! warning "Warning"
            Since the maximum diversification objective is generally non-convex, SciPy's `differential_evolution` optimizer
            is used to obtain more robust solutions. This approach incurs significantly higher computational cost and should
            be used with care.

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
            `custom_cov` (*None or array-like of shape (n_assets, n_assets), optional*): Custom covariance matrix. Can be used to inject externally generated covariance matrices (eg. Ledoit-Wolf). Defaults to `None`.
            seed (*int or None, optional*): Seed for differential evolution solver. Defaults to `100` to preserve deterministic outputs.
            `**kwargs` (*optional*): Included for interface consistency, allowing the backtesting engine to pass additional or optimizer-specific arguments that may be safely ignored by this optimizer.

        **Returns:**

        - `np.ndarray`: Vector of optimized portfolio weights.

        Raises:
            DataError: For any data mismatch during integrity check.
            OptimizationError: If `differential_evolution` solver fails to solve.

        !!! example "Example:"
            ```python
            # Importing the maximum diversification module
            from opes.objectives import MaxDiversification

            # Let this be your ticker data
            training_data = some_data()

            # Let this be your custom covariance matrix
            cov_m = customCov()

            # Initialize with custom regularization
            maxdiv = MaxDiversification(reg='entropy', strength=0.01)

            # Optimize portfolio with custom seed and covariance matrix
            weights = maxdiv.optimize(data=training_data, custom_cov=cov_m, seed=46)
            ```
        """
        # Preparing optimization and finding constraint
        self._prepare_optimization_inputs(data, custom_cov=custom_cov)

        # Optimization objective and results
        def f(w):
            w = w / (w.sum() + 1e-10)
            var = w @ self.covariance @ w + 1e-10
            weightvol = w @ self.volarray
            return -(weightvol / np.sqrt(var)) + self.strength * self.reg(w)

        result = differential_evolution(
            f,
            strategy="randtobest1bin",
            bounds=[(0, 1) for _ in range(len(self.weights))],
            rng=seed,
        )
        if result.success:
            self.weights = result.x / (result.x.sum() + 1e-12)
            return self.weights.copy()
        else:
            raise OptimizationError(
                f"Maximum diversification optimization failed: {result.message}"
            )

    def set_regularizer(self, reg=None, strength=1):
        """
        Updates the regularization function and its penalty strength.

        This method updates both the regularization function and its associated
        penalty strength. Useful for changing the behaviour of the optimizer without
        initiating a new one.

        Args:
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.

        !!! example "Example:"
            ```python
            # Import the MaxDiversification class
            from opes.objectives import MaxDiversification

            # Set with 'entropy' regularization
            optimizer = MaxDiversification(reg='entropy', strength=0.01)

            # --- Do Something with `optimizer` ---
            optimizer.optimize(data=some_data())

            # Change regularizer using `set_regularizer`
            optimizer.set_regularizer(reg='l1', strength=0.02)

            # --- Do something else with new `optimizer` ---
            optimizer.optimize(data=some_data())
            ```
        """
        self.reg = _find_regularizer(reg)
        self.strength = strength


class RiskParity(Optimizer):
    """
    Equal Risk Contribution (Risk Parity) optimization.

    Risk parity, developed and popularized by Edward Qian and others in the
    1990s-2000s, allocates capital such that each asset contributes equally
    to total portfolio risk. The core insight is that market-capitalization
    or equal-weight portfolios are dominated by the risk of a few volatile
    assets (typically equities), leaving other assets (bonds, commodities)
    with minimal risk contribution. Risk parity addresses this by leveraging
    low-volatility assets and de-leveraging high-volatility ones to equalize
    risk contributions.
    """

    def __init__(self, reg=None, strength=1):
        """
        Args:
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "riskparity"
        self.reg = _find_regularizer(reg)
        self.strength = strength
        self.covariance = None

        self.tickers = None
        self.weights = None

    def _prepare_optimization_inputs(self, data, weight_bounds, w, custom_cov=None):
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)

        # Checking for covariance and weights
        if custom_cov is None:
            # Handling invertibility using the small epsilon * identity matrix
            # small epsilon scales with the trace of the covariance
            self.covariance = np.cov(data, rowvar=False)
            epsilon = 1e-3 * np.trace(self.covariance) / self.covariance.shape[0]
            self.covariance = self.covariance + epsilon * np.eye(
                self.covariance.shape[0]
            )
        else:
            self.covariance = custom_cov
        self.weights = np.array(
            np.ones(len(self.tickers)) / len(self.tickers) if w is None else w,
            dtype=float,
        )

        # Functions to test data integrity
        test_integrity(
            tickers=self.tickers,
            weights=self.weights,
            cov=self.covariance,
            bounds=weight_bounds,
        )

    def optimize(self, data=None, weight_bounds=(0, 1), w=None, custom_cov=None):
        """
        Solves the Risk Parity objective (Target Contribution Variant):

        $$
        \\min_{\\mathbf{w}} \\ \\sum_i^N \\left(RC_i - TC\\right)^2
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
            weight_bounds (*tuple, optional*): Boundary constraints for asset weights. Values must be of the format `(lesser, greater)` with `0 <= |lesser|, |greater| <= 1`. Defaults to `(0,1)`.
            w (*None or np.ndarray, optional*): Initial weight vector for warm starts. Mainly used for backtesting and not recommended for user input. Defaults to `None`.
            `custom_cov` (*None or array-like of shape (n_assets, n_assets), optional*): Custom covariance matrix. Can be used to inject externally generated covariance matrices (eg. Ledoit-Wolf). Defaults to `None`.

        **Returns:**

        - `np.ndarray`: Vector of optimized portfolio weights.

        Raises:
            DataError: For any data mismatch during integrity check.
            OptimizationError: If `SLSQP` solver fails to solve.

        !!! example "Example:"
            ```python
            # Importing the risk parity module
            from opes.objectives import RiskParity

            # Let this be your ticker data
            training_data = some_data()

            # Let this be your custom covariance
            cov_m = customCov()

            # Initialize with custom regularization
            rp = RiskParity(reg='entropy', strength=0.01)

            # Optimize portfolio with custom weight bounds and covariance matrix
            weights = rp.optimize(data=training_data, weight_bounds=(0.05, 0.75), custom_cov=cov_m)
            ```
        """
        # Preparing optimization and finding constraint
        self._prepare_optimization_inputs(data, weight_bounds, w, custom_cov=custom_cov)
        constraint = find_constraint(weight_bounds)
        w = self.weights

        # Optimization objective and results
        def f(w):
            portfolio_volatility = max(np.sqrt((w @ self.covariance @ w)), 1e-10)
            risk_contribution = w * (self.covariance @ w) / portfolio_volatility
            target_contribution = portfolio_volatility / len(w)
            return np.sum(
                (risk_contribution - target_contribution) ** 2
            ) + self.strength * self.reg(w)

        result = minimize(
            f,
            w,
            method="SLSQP",
            bounds=[weight_bounds] * len(w),
            constraints=constraint,
        )
        if result.success:
            self.weights = result.x
            return self.weights.copy()
        else:
            raise OptimizationError(
                f"Risk parity optimization failed: {result.message}"
            )

    def set_regularizer(self, reg=None, strength=1):
        """
        Updates the regularization function and its penalty strength.

        This method updates both the regularization function and its associated
        penalty strength. Useful for changing the behaviour of the optimizer without
        initiating a new one.

        Args:
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.

        !!! example "Example:"
            ```python
            # Import the RiskParity class
            from opes.objectives import RiskParity

            # Set with 'entropy' regularization
            optimizer = RiskParity(reg='entropy', strength=0.01)

            # --- Do Something with `optimizer` ---
            optimizer.optimize(data=some_data())

            # Change regularizer using `set_regularizer`
            optimizer.set_regularizer(reg='l1', strength=0.02)

            # --- Do something else with new `optimizer` ---
            optimizer.optimize(data=some_data())
            ```
        """
        self.reg = _find_regularizer(reg)
        self.strength = strength


class REPO(Optimizer):
    """
    Return-Entropy Portfolio Optimization (REPO).

    Return-Entropy Portfolio Optimization (REPO), introduced by Mercurio et. al.,
    applies Shannon entropy as a risk measure for portfolios of continuous-return assets.
    The method addresses five key limitations of Markowitz's mean-variance portfolio
    optimization: tendency toward sparse solutions with large weights on high-risk assets,
    disturbance of asset dependence structures when using investor views, instability of
    optimal solutions under input adjustments, difficulty handling non-normal or asymmetric
    return distributions, and challenges in estimating covariance matrices. By using entropy
    rather than variance as the risk measure, REPO naturally accommodates asymmetric distributions.
    """

    def __init__(self, risk_aversion=1, reg=None, strength=1):
        """
        Args:
            risk_aversion (*float, optional*): Risk-aversion coefficient. Higher values emphasize risk (entropy) minimization, while lower values favor return seeking. Usually greater than `0`. Defaults to `0.5`.
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "repo"
        self.reg = _find_regularizer(reg)
        self.strength = strength
        self.risk_aversion = risk_aversion
        self.mean = None

        self.tickers = None
        self.weights = None

    def _prepare_optimization_inputs(self, data, bins, custom_mean=None):
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(
            np.ones(len(self.tickers)) / len(self.tickers), dtype=float
        )
        self.mean = np.mean(data, axis=0) if custom_mean is None else custom_mean

        # Functions to test data integrity and find optimization constraint
        test_integrity(
            tickers=self.tickers,
            weights=self.weights,
            mean=self.mean,
            hist_bins=bins,
        )
        return data

    def optimize(self, data=None, bin=20, custom_mean=None, seed=100, **kwargs):
        """
        Solves the Return-Entropy-Portfolio-Optimization objective:

        $$
        \\min_{\\mathbf{w}} \\ \\gamma \\mathcal{H}(\\mathbf{r}) - \\mathbf{w}^\\top \\mu + \\lambda R(\\mathbf{w})
        $$

        !!! warning "Warning"
            Since REPO objective is generally non-convex, SciPy's `differential_evolution` optimizer
            is used to obtain more robust solutions. This approach incurs significantly higher computational cost and should
            be used with care.

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
            bin (*int, optional*): Number of histogram bins to be used for the return distribution. Defaults to `20`.
            custom_mean (*None or np.ndarray, optional*): Custom mean vector. Can be used to inject externally generated mean vectors (eg. Black-Litterman). Defaults to `None`.
            seed (*int or None, optional*): Seed for differential evolution solver. Defaults to `100` to preserve deterministic outputs.
            `**kwargs` (*optional*): Included for interface consistency, allowing the backtesting engine to pass additional or optimizer-specific arguments that may be safely ignored by this optimizer.

        **Returns:**

        - `np.ndarray`: Vector of optimized portfolio weights.

        Raises:
            DataError: For any data mismatch during integrity check.
            OptimizationError: If `differential_evolution` solver fails to solve.

        !!! example "Example:"
            ```python
            # Importing the REPO module
            from opes.objectives import REPO

            # Let this be your ticker data
            training_data = some_data()

            # Let this be your custom mean vector
            mean_v = customMean()

            # Initialize with custom regularization
            repo = REPO(reg='entropy', strength=0.01)

            # Optimize portfolio with custom seed, bins and mean vector
            weights = repo.optimize(data=training_data, bin=15, custom_mean=mean_v, seed=46)
            ```
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self._prepare_optimization_inputs(
            data, bin, custom_mean=custom_mean
        )

        # Optimization objective and results
        def f(w):
            w = w / w.sum()
            X = trimmed_return_data @ w
            # Constructing histogram
            counts = np.histogram(X, bins=bin, density=False)[0]
            probabilities = counts / counts.sum()
            probabilities = probabilities[probabilities > 0]
            # Computing both the terms
            mean_term = self.mean @ w
            entropy_term = -np.sum(probabilities * np.log(probabilities))
            return (
                -mean_term
                + self.risk_aversion * entropy_term
                + self.strength * self.reg(w)
            )

        result = differential_evolution(
            f,
            strategy="randtobest1bin",
            bounds=[(0, 1) for _ in range(len(self.weights))],
            rng=seed,
        )
        if result.success:
            self.weights = result.x / (result.x.sum() + 1e-12)
            return self.weights.copy()
        else:
            raise OptimizationError(f"REPO optimization failed: {result.message}")

    def set_regularizer(self, reg=None, strength=1):
        """
        Updates the regularization function and its penalty strength.

        This method updates both the regularization function and its associated
        penalty strength. Useful for changing the behaviour of the optimizer without
        initiating a new one.

        Args:
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.

        !!! example "Example:"
            ```python
            # Import the REPO class
            from opes.objectives import REPO

            # Set with 'entropy' regularization
            optimizer = REPO(reg='entropy', strength=0.01)

            # --- Do Something with `optimizer` ---
            optimizer.optimize(data=some_data())

            # Change regularizer using `set_regularizer`
            optimizer.set_regularizer(reg='l1', strength=0.02)

            # --- Do something else with new `optimizer` ---
            optimizer.optimize(data=some_data())
            ```
        """
        self.reg = _find_regularizer(reg)
        self.strength = strength


class HierarchicalRiskParity(Optimizer):
    """
    Hierarchical Risk Parity (HRP) optimization.

    Hierarchical Risk Parity (HRP), introduced by López de Prado,
    is a portfolio construction methodology that allocates capital
    through hierarchical clustering and recursive risk balancing
    rather than direct optimization of a scalar objective. HRP
    addresses several structural weaknesses of traditional
    mean-variance and risk parity approaches, including sensitivity
    to covariance matrix estimation error, numerical instability
    arising from matrix inversion and over-concentration in highly
    correlated assets. By organizing assets into a hierarchy based
    on correlation structure and allocating weights inversely to
    cluster-level variance, HRP achieves diversification across
    both individual assets and correlated groups. This procedure
    yields stable, fully invested portfolios that are robust
    out-of-sample and well-suited for high-dimensional or noisy
    return environments, particularly when covariance estimates are
    ill-conditioned.
    """

    def __init__(self, cluster_method="average"):
        """
        **Args:**

        - `cluster_method` (*str, optional*): Clustering method to be used for hierarchical clustering. Defaults to `'average'`. Available methods are:
            - '`average`': Merges clusters based on the minimum pairwise distance, often producing elongated, chain-like clusters.
            - '`single'`: Merges clusters based on the maximum pairwise distance, favoring compact and tightly bound clusters.
            - `'complete'`: Merges clusters using the average pairwise distance between all points in each cluster, providing a balanced compromise between single and complete linkage.
            - '`ward`': Merges clusters to minimize the increase in within-cluster variance, producing balanced clusters aligned with variance-based portfolio construction.
        """
        self.cluster_method = cluster_method.lower()
        self.covariance = None
        self.correlation = None

        self.weights = None
        self.tickers = None

    # Helper method to prepare inputs for optimization
    # Also validates parameters and checks for data inconsistency
    def _prepare_optimization_inputs(self, data, custom_cov):

        # Checking for cluster method validity
        if self.cluster_method not in ["single", "average", "complete", "ward"]:
            raise PortfolioError(f"Unknown cluster_method: {self.cluster_method}")

        # Obtaining return and ticker data
        self.tickers, data = extract_trim(data)

        # Computing correlation matrix
        self.correlation = np.corrcoef(data, rowvar=False)

        # Checking for covariance and weights and assigning optimization data accordingly
        if custom_cov is None:
            # Invertibility need not be handled as usual since HRP does not invert covariance
            self.covariance = np.cov(data, rowvar=False)
        else:
            self.covariance = custom_cov
        # We start with all assets set to 1
        # This is different from the usual equal weight since HRP allocates step by step
        # But is initialized as 1 before allocation
        self.weights = np.ones(len(self.tickers))

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, cov=self.covariance)

    # Function to get leaf order of dendogram
    def _get_leaf_order(self, condensed_distance_matrix):
        clustered_linkage = linkage(
            condensed_distance_matrix, method=self.cluster_method
        )
        cluster_dendogram = dendrogram(clustered_linkage, no_plot=True)
        return np.array(cluster_dendogram["leaves"])

    # Function to get cluster variance
    def _get_cluster_variance(self, cluster):
        # Slicing covariance which is required for the cluster
        cov_slice = self.covariance[np.ix_(cluster, cluster)]

        # Finding variance (diagonal elements)
        diag = np.diag(cov_slice)

        # Computing cluster variance
        # NOTE: 1e-12 added for numerical stability
        inv_diag = 1.0 / (diag + 1e-12)
        return 1.0 / inv_diag.sum()

    # 'Recursive' bisection function
    # NOTE: while loop + queue is used for stability
    def _recursive_bisection(self, leaf_order):
        clusters = [leaf_order]

        # Instead of recursion, which might hit a stack limit in python, we use a while loop and a queue
        # Not a queue exactly, but we similate one using pop(0) and append()
        while clusters:
            cluster = clusters.pop(0)

            # Checking if cluster is a single element or none
            if len(cluster) <= 1:
                continue

            # Splitting clusters into left and right subsections
            split_var = len(cluster) // 2
            left_cluster = cluster[:split_var]
            right_cluster = cluster[split_var:]

            # Obtaining cluster variance for left and right
            left_variance = self._get_cluster_variance(left_cluster)
            right_variance = self._get_cluster_variance(right_cluster)

            # Computing left weights (alpha)
            alpha = 1 - left_variance / (left_variance + right_variance)

            # Assigning weights to left and right clusters respectively
            self.weights[left_cluster] *= alpha
            self.weights[right_cluster] *= 1 - alpha

            # Appending for recursion
            clusters.append(left_cluster)
            clusters.append(right_cluster)

        self.weights = self.weights / self.weights.sum()

    def optimize(self, data, custom_cov=None):
        """
        Computes the Hierarchical Risk Parity portfolio:

        $$
        \\mathbf{w}_i \\propto \\frac{1}{\\sigma^2_{\\text{cluster}}}
        $$

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
            `custom_cov` (*None or array-like of shape (n_assets, n_assets), optional*): Custom covariance matrix. Can be used to inject externally generated covariance matrices (eg. Ledoit-Wolf). Defaults to `None`.

        **Returns:**

        - `np.ndarray`: Vector of optimized portfolio weights.

        Raises:
            DataError: For any data mismatch during integrity check.
            PortfolioError: For any invalid portfolio variable inputs during integrity check.

        !!! example "Example:"
            ```python
            # Importing the HRP module
            from opes.objectives import HierarchicalRiskParity as HRP

            # Let this be your ticker data
            training_data = some_data()

            # Let this be your custom covariance matrix
            cov_m = covMatrix()

            # Initialize with custom clustering method
            hrp_portfolio = HRP(cluster_method='ward')

            # Optimize portfolio with custom covariance matrix
            weights = hrp_portfolio.optimize(data=training_data, custom_cov=cov_m)
            ```
        """
        # Preparing inputs for HRP optimization
        self._prepare_optimization_inputs(data, custom_cov=custom_cov)

        # Computing distance matrix and condensing it for leaf order
        distance_matrix = np.sqrt(0.5 * (1 - self.correlation))
        condensed_distance = squareform(distance_matrix, checks=False)
        leaf_order = self._get_leaf_order(condensed_distance)

        self._recursive_bisection(leaf_order)
        return self.weights.copy()
