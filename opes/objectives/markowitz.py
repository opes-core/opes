"""
Markowitz Portfolio Theory (MPT), introduced by Harry Markowitz in his 1952
paper *Portfolio Selection*, established a formal mathematical approach to portfolio
construction by shifting the focus from individual securities to the risk-return
characteristics of the portfolio as a whole. Its core insight is that diversification
reduces risk when asset returns are not perfectly correlated, allowing investors to
achieve either higher expected returns for a given level of risk or lower risk for a
given expected return, a contribution later recognized with the 1990 Nobel Prize in Economics.

---
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

from opes.objectives.base_optimizer import Optimizer
from ..regularizer import _find_regularizer
from ..utils import extract_trim, test_integrity, find_constraint
from ..errors import OptimizationError


class MaxMean(Optimizer):
    """
    Maximum Mean return optimization.

    Maximum mean represents the limiting case of Markowitz's framework where
    the investor is risk-neutral and cares only about expected return. This
    portfolio allocates capital entirely to the asset(s) with the highest
    expected return, ignoring risk considerations. Without short-sale constraints,
    this strategy places 100% weight in the single asset with maximum expected return.
    While theoretically simple, this approach is rarely used in practice as it
    completely ignores diversification benefits and risk management.
    """

    def __init__(self, reg=None, strength=1):
        """
        Args:
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "maxmean"
        self.reg = _find_regularizer(reg)
        self.strength = strength
        self.mean = None

        self.tickers = None
        self.weights = None

    def _prepare_optimization_inputs(self, data, weight_bounds, w, custom_mean=None):
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)

        # Checking for mean and weights and assigning optimization data accordingly
        self.mean = np.mean(data, axis=0) if custom_mean is None else custom_mean
        self.weights = np.array(
            np.ones(len(self.tickers)) / len(self.tickers) if w is None else w,
            dtype=float,
        )

        # Functions to test data integrity and find optimization constraint
        test_integrity(
            tickers=self.tickers,
            weights=self.weights,
            mean=self.mean,
            bounds=weight_bounds,
        )

    def optimize(self, data=None, weight_bounds=(0, 1), w=None, custom_mean=None):
        """
        Solves the Maximum Mean objective:

        $$
        \\min_{\\mathbf{w}} \\ - \\mathbf{w}^\\top \\mu + \\lambda R(\\mathbf{w})
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
            custom_mean (*None or np.ndarray, optional*): Custom mean vector. Can be used to inject externally generated mean vectors (eg. Black-Litterman). Defaults to `None`.

        **Returns:**

        - `np.ndarray`: Vector of optimized portfolio weights.

        Raises:
            DataError: For any data mismatch during integrity check.
            PortfolioError: For any invalid portfolio variable inputs during integrity check.
            OptimizationError: If `SLSQP` solver fails to solve.

        !!! example "Example:"
            ```python
            # Importing the maximum mean module
            from opes.objectives.markowitz import MaxMean

            # Let this be your ticker data
            training_data = some_data()

            # Let this be your custom mean vector
            # Can be Black-Litterman, Bayesian, Fama-French etc.
            mean_v = customMean()

            # Initialize with custom regularization
            maxmean = MaxMean(reg='entropy', strength=0.02)

            # Optimize portfolio with custom weight bounds and custom mean vector
            weights = maxmean.optimize(data=training_data, weight_bounds=(0.05, 0.75), custom_mean=mean_v)
            ```
        """
        # Preparing optimization and finding constraint
        self._prepare_optimization_inputs(
            data, weight_bounds, w, custom_mean=custom_mean
        )
        constraint = find_constraint(weight_bounds)
        w = self.weights

        # Optimization objective and results
        def f(w):
            return -(self.mean @ w - self.strength * self.reg(w))

        result = minimize(
            f,
            w,
            method="SLSQP",
            bounds=[weight_bounds] * len(w),
            constraints=constraint,
        )
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(
                f"Maximum mean optimization failed: {result.message}"
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
            # Import the MaxMean class
            from opes.objectives.markowitz import MaxMean

            # Set with 'entropy' regularization
            optimizer = MaxMean(reg='entropy', strength=0.01)

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


class MinVariance(Optimizer):
    """
    Minimum Variance optimization.

    The Global Minimum Variance portfolio represents the
    lowest-risk portfolio achievable through diversification,
    regardless of expected returns. This portfolio lies at the leftmost
    point of the efficient frontier and has the important property that
    it does not require estimates of expected returns. This makes GMV
    portfolios more robust to estimation error than other mean-variance
    strategies, as expected returns are notoriously difficult to estimate
    accurately, even more so than covariance matrix. The GMV portfolio is
    particularly popular among practitioners who are skeptical of return
    forecasts or who seek a purely defensive allocation.
    """

    def __init__(self, reg=None, strength=1):
        """
        Args:
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "gmv"
        self.reg = _find_regularizer(reg)
        self.strength = strength
        self.covariance = None

        self.tickers = None
        self.weights = None

    def _prepare_optimization_inputs(self, data, weight_bounds, w, custom_cov=None):
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)

        # Checking for covariance and weights and assigning optimization data accordingly
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

        # Functions to test data integrity and find optimization constraint
        test_integrity(
            tickers=self.tickers,
            weights=self.weights,
            cov=self.covariance,
            bounds=weight_bounds,
        )

    def optimize(self, data=None, weight_bounds=(0, 1), w=None, custom_cov=None):
        """
        Solves the Minimum Variance objective:

        $$
        \\min_{\\mathbf{w}} \\ \\mathbf{w}^\\top \\Sigma \\mathbf{w} + \\lambda R(\\mathbf{w})
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
            PortfolioError: For any invalid portfolio variable inputs during integrity check.
            OptimizationError: If `SLSQP` solver fails to solve.

        !!! example "Example:"
            ```python
            # Importing the Global Minimum Variance (GMV) module
            from opes.objectives.markowitz import MinVariance

            # Let this be your ticker data
            training_data = some_data()

            # Let this be your custom covariance matrix
            # Can be Ledoit-Wolf, OAS, BARRA etc.
            cov_m = customCov()

            # Initialize GMV optimizer with custom regularizer
            gmv = MinVariance(reg='entropy', strength=0.05)

            # Optimize portfolio with custom weight bounds and covariance matrix
            weights_ftrl = gmv.optimize(data=training_data, weight_bounds=(0.05, 0.8), custom_cov=cov_m)
            ```
        """
        # Preparing optimization and finding constraint
        self._prepare_optimization_inputs(data, weight_bounds, w, custom_cov=custom_cov)
        constraint = find_constraint(weight_bounds)
        w = self.weights

        # Optimization objective and results
        def f(w):
            return w @ self.covariance @ w + self.strength * self.reg(w)

        result = minimize(
            f,
            w,
            method="SLSQP",
            bounds=[weight_bounds] * len(w),
            constraints=constraint,
        )
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(
                f"Global minimum optimization failed: {result.message}"
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
            # Import the MinVariance class
            from opes.objectives.markowitz import MinVariance

            # Set with 'entropy' regularization
            optimizer = MinVariance(reg='entropy', strength=0.01)

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


class MeanVariance(Optimizer):
    """
    Mean-Variance Optimization.

    Mean-variance optimization trades off expected return and risk
    via a risk-aversion parameter, where higher values produce more
    conservative portfolios closer to the global minimum variance
    solution and lower values lead to more aggressive, return-seeking
    allocations.
    """

    def __init__(self, risk_aversion=0.5, reg=None, strength=1):
        """
        Args:
            risk_aversion (*float, optional*): Risk-aversion coefficient. Higher values emphasize risk minimization, while lower values favor return seeking. Usually greater than `0`. Defaults to `0.5`.
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "mvo"
        self.reg = _find_regularizer(reg)
        self.risk_aversion = risk_aversion
        self.strength = strength
        self.covariance = None
        self.mean = None

        self.tickers = None
        self.weights = None

    def _prepare_optimization_inputs(
        self, data, weight_bounds, w, custom_cov=None, custom_mean=None
    ):
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)

        # Checking for mean, covaraince and weights and assigning optimization data accordingly
        self.mean = np.mean(data, axis=0) if custom_mean is None else custom_mean
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

        # Functions to test data integrity and find optimization constraint
        test_integrity(
            tickers=self.tickers,
            weights=self.weights,
            cov=self.covariance,
            bounds=weight_bounds,
        )

    def optimize(
        self, data=None, weight_bounds=(0, 1), w=None, custom_cov=None, custom_mean=None
    ):
        """
        Solves the Mean-Variance Optimization objective:

        $$
        \\min_{\\mathbf{w}} \\ \\frac{\\gamma}{2} \\mathbf{w}^\\top \\Sigma \\mathbf{w} - \\mathbf{w}^\\top \\mu + \\lambda R(\\mathbf{w})
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
            custom_mean (*None or np.ndarray, optional*): Custom mean vector. Can be used to inject externally generated mean vectors (eg. Black-Litterman). Defaults to `None`.
            `custom_cov` (*None or array-like of shape (n_assets, n_assets), optional*): Custom covariance matrix. Can be used to inject externally generated covariance matrices (eg. Ledoit-Wolf). Defaults to `None`.

        **Returns:**

        - `np.ndarray`: Vector of optimized portfolio weights.

        Raises:
            DataError: For any data mismatch during integrity check.
            PortfolioError: For any invalid portfolio variable inputs during integrity check.
            OptimizationError: If `SLSQP` solver fails to solve.

        !!! example "Example:"
            ```python
            # Importing the mean variance module
            from opes.objectives.markowitz import MeanVariance

            # Let this be your ticker data
            training_data = some_data()

            # Let these be your custom mean and covariance
            mean_v = customMean()
            cov_m = customCov()

            # Initialize with risk aversion and custom regularization
            mean_variance = MeanVariance(risk_aversion=0.33, reg='entropy', strength=0.01)

            # Optimize portfolio with custom weight bounds, mean vector and covariance matrix
            weights = mean_variance.optimize(data=training_data, weight_bounds=(0.05, 0.75), custom_mean=mean_v, custom_cov=cov_m)
            ```
        """
        # Preparing optimization and finding constraint
        self._prepare_optimization_inputs(
            data, weight_bounds, w, custom_cov=custom_cov, custom_mean=custom_mean
        )
        constraint = find_constraint(weight_bounds)
        w = self.weights

        # Optimization objective and results
        def f(w):
            return (
                -self.mean @ w
                + (self.risk_aversion / 2) * (w @ self.covariance @ w)
                + self.strength * self.reg(w)
            )

        result = minimize(
            f,
            w,
            method="SLSQP",
            bounds=[weight_bounds] * len(w),
            constraints=constraint,
        )
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(
                f"Mean variance optimization failed: {result.message}"
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
            # Import the MeanVariance class
            from opes.objectives.markowitz import MeanVariance

            # Set with 'entropy' regularization
            optimizer = MeanVariance(reg='entropy', strength=0.01)

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


class MaxSharpe(Optimizer):
    """
    Maximum Sharpe Optimization.

    The maximum Sharpe ratio portfolio, formalized by William F. Sharpe,
    follows directly from Markowitz's framework. Also known as the tangency
    portfolio, this portfolio provides the highest risk-adjusted return as
    measured by excess return per unit of risk. When a risk-free asset is
    available, this portfolio represents the optimal risky portfolio for all
    mean-variance investors regardless of their risk aversion.
    """

    def __init__(self, risk_free=0.01, reg=None, strength=1):
        """
        Args:
            risk_free (*float, optional*): Risk-free rate. A non-zero value induces the tangency portfolio. Defaults to `0.01`.
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "maxsharpe"
        self.reg = _find_regularizer(reg)
        self.risk_free = risk_free
        self.strength = strength
        self.covariance = None
        self.mean = None

        self.tickers = None
        self.weights = None

    def _prepare_optimization_inputs(self, data, custom_cov=None, custom_mean=None):
        # Extracting trimmed return data from OHLCV and obtaining tickers
        self.tickers, data = extract_trim(data)

        # Checking for mean, covariance and weights and assigning optimization data accordingly
        self.mean = np.mean(data, axis=0) if custom_mean is None else custom_mean
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
            np.ones(len(self.tickers)) / len(self.tickers), dtype=float
        )

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, cov=self.covariance)

    def optimize(
        self, data=None, custom_cov=None, custom_mean=None, seed=100, **kwargs
    ):
        """
        Solves the Maximum Sharpe objective:

        $$
        \\min_{\\mathbf{w}} \\ -\\frac{\\mathbf{w}^\\top \\mu - r_f}{\\sqrt{\\mathbf{w}^\\top \\Sigma \\mathbf{w}}} + \\lambda R(\\mathbf{w})
        $$

        !!! warning "Warning"
            Since the maximum Sharpe objective is generally non-convex, SciPy's `differential_evolution` optimizer is used to obtain more robust solutions.
            This approach incurs significantly higher computational cost and should be used with care.

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
            `custom_cov` (*None or array-like of shape (n_assets, n_assets), optional*): Custom covariance matrix. Can be used to inject externally generated covariance matrices (eg. Ledoit-Wolf). Defaults to `None`.
            seed (*int or None, optional*): Seed for differential evolution solver. Defaults to `100` to preserve deterministic outputs.
            **kwargs (*optional*): Included for interface consistency, allowing the backtesting engine to pass additional or optimizer-specific arguments that may be safely ignored by this optimizer.

        **Returns:**

        - `np.ndarray`: Vector of optimized portfolio weights.

        Raises:
            DataError: For any data mismatch during integrity check.
            PortfolioError: For any invalid portfolio variable inputs during integrity check.
            OptimizationError: If `differential_evolution` solver fails to solve.

        !!! example "Example:"
            ```python
            # Importing the maximum sharpe module
            from opes.objectives.markowitz import MaxSharpe

            # Let this be your ticker data
            training_data = some_data()

            # Let these be your custom mean and covariance
            mean_v = customMean()
            cov_m = customCov()

            # Initialize with risk free rate and custom regularization
            maxsharpe = MaxSharpe(risk_free=0.02, reg='entropy', strength=0.01)

            # Optimize portfolio with custom  mean vector and covariance matrix
            # We also set the seed to 46
            weights = maxsharpe.optimize(data=training_data, custom_mean=mean_v, custom_cov=cov_m, seed=46)
            ```
        """
        # Preparing optimization and finding constraint
        self._prepare_optimization_inputs(
            data, custom_cov=custom_cov, custom_mean=custom_mean
        )

        # Optimization objective and results
        def f(w):
            w = w / (w.sum() + 1e-10)
            return -(
                (self.mean @ w - self.risk_free)
                / max(np.sqrt((w @ self.covariance @ w)), 1e-10)
                - self.strength * self.reg(w)
            )

        result = differential_evolution(
            f,
            strategy="randtobest1bin",
            bounds=[(0, 1) for _ in range(len(self.weights))],
            rng=seed,
        )
        if result.success:
            self.weights = result.x / (result.x.sum() + 1e-12)
            return self.weights
        else:
            raise OptimizationError(
                f"Maximum sharpe optimization failed: {result.message}"
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
            # Import the MaxSharpe class
            from opes.objectives.markowitz import MaxSharpe

            # Set with 'entropy' regularization
            optimizer = MaxSharpe(reg='entropy', strength=0.01)

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
