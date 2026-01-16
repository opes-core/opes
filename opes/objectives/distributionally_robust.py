"""
Distributionally robust optimization (DRO) addresses a fundamental challenge in portfolio management:
the true distribution of asset returns is unknown and estimates from historical data are subject to
sampling error. Rather than optimizing with respect to a single estimated distribution which can lead
to severe out-of-sample underperformance when the estimate is poor, DRO optimizes against the worst-case
distribution within an ambiguity set of plausible distributions centered around an empirical or reference
distribution.

OPES leverages dual formulations of selected DRO variants, reducing the resulting problems to tractable,
and in many cases, convex minimization programs. This includes formulations drawn from both established
literature and recent, cutting-edge work, some of which may not yet be fully peer-reviewed. While several
dual representations are well-known (e.g. the Kantorovich-Rubinstein duality), others remain comparatively
less visible in the existing literature.

---
"""

from numbers import Real
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opes.objectives.base_optimizer import Optimizer
from ..utils import extract_trim, test_integrity, find_constraint
from ..errors import OptimizationError, PortfolioError


class KLRobustMaxMean(Optimizer):
    """
    Kullback-Leibler Ambiguity Maximum Mean optimization.

    Optimizes the expected return under the worst-case probability distribution
    within a KL-divergence uncertainty ball (radius) around the empirical distribution. This problem was
    analyzed by Hu and Hong in their comprehensive study of KL-constrained distributionally robust
    optimization, who showed it admits a tractable convex reformulation through Fenchel duality
    and change-of-measure techniques.
    """

    def __init__(self, radius=0.01):
        """
        Args:
            radius (*float, optional*): The size of the uncertainty set (KL-divergence bound). Larger values indicate higher uncertainty. Defaults to `0.01`.
        """
        self.identity = "kldr-mm"
        self.radius = radius

        self.tickers = None
        self.weights = None

    def _prepare_optimization_inputs(self, data, weight_bounds, w):
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(
            np.ones(len(self.tickers)) / len(self.tickers) if w is None else w,
            dtype=float,
        )

        # Functions to test data integrity and find optimization constraint
        test_integrity(
            tickers=self.tickers,
            weights=self.weights,
            bounds=weight_bounds,
            uncertainty_radius=self.radius,
        )
        return data

    def optimize(self, data=None, weight_bounds=(0, 1), w=None):
        """
        Solves the KL-maximum-mean dual objective:

        $$
        \\min_{\\mathbf{w}, \\alpha \\ge 0} \\ \\alpha \\log \\mathbb{E}_{\\mathbb{P}} \\left[e^{\\mathbf{w}^\\top \\mathbf{r} / \\alpha}\\right] + \\alpha \\epsilon
        $$

        Uses the log-sum-exp technique to solve for numerical stability.

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

        **Returns:**

        - `np.ndarray`: Vector of optimized portfolio weights.

        Raises:
            DataError: For any data mismatch during integrity check.
            PortfolioError: For any invalid portfolio variable inputs during integrity check.
            OptimizationError: If `SLSQP` solver fails to solve.

        !!! example "Example:"
            ```python
            # Importing the dro maximum mean module
            from opes.objectives.distributionally_robust import KLRobustMaxMean

            # Let this be your ticker data
            training_data = some_data()

            # Initialize with KL divergence radius
            kl_maxmean = KLRobustMaxMean(radius=0.02)

            # Optimize portfolio with custom weight bounds
            weights = kl_maxmean.optimize(data=training_data, weight_bounds=(0.05, 0.75))
            ```
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self._prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds, constraint_type=2)
        param_array = np.append(self.weights, 1)

        # Optimization objective and results
        def f(x):
            w, dual_var = x[:-1], x[-1]
            X = -(trimmed_return_data @ w / dual_var)
            # Utilize the log-sum-exp tecnique to ensure numerical stability
            m = np.max(X)
            return dual_var * self.radius + dual_var * (
                m + np.log(np.mean(np.exp(X - m)))
            )

        result = minimize(
            f,
            param_array,
            method="SLSQP",
            bounds=[weight_bounds] * len(self.weights) + [(1e-3, None)],
            constraints=constraint,
        )
        if result.success:
            self.weights = result.x[:-1]
            return self.weights
        else:
            raise OptimizationError(
                f"KL robust maximum mean optimization failed: {result.message}"
            )


class KLRobustKelly(Optimizer):
    """
    Kullback-Leibler Ambiguity Kelly Criterion.

    Maximizes the expected logarithmic wealth under the worst-case probability
    distribution within a specified KL-divergence radius. The distributionally
    robust Kelly criterion addresses estimation error in growth-optimal portfolios
    by maximizing expected log growth against worst-case distributions within a
    KL ambiguity set. The KL-robust Kelly criterion produces portfolios that are
    more diversified than the standard Kelly portfolio, trading off some growth
    rate for robustness against distributional uncertainty.
    """

    def __init__(self, fraction=1.0, radius=0.01):
        """
        Args:
            radius (*float, optional*): The size of the uncertainty set (KL-divergence bound). Larger values indicate higher uncertainty. Defaults to `0.01`.
            fraction (*float, optional*): kelly fractional exposure to the market. Must be within (0,1]. Lower values bet less aggressively. Defaults to `1.0`.
        """
        self.identity = "kldr-kelly"
        self.radius = radius
        self.fraction = fraction

        self.tickers = None
        self.weights = None

    def _prepare_optimization_inputs(self, data, weight_bounds, w):
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(
            np.ones(len(self.tickers)) / len(self.tickers) if w is None else w,
            dtype=float,
        )

        # Functions to test data integrity and find optimization constraint
        test_integrity(
            tickers=self.tickers,
            weights=self.weights,
            bounds=weight_bounds,
            uncertainty_radius=self.radius,
        )
        return data

    def optimize(self, data=None, weight_bounds=(0, 1), w=None):
        """
        Solves the Hu and Hong KL-Kelly dual objective:

        $$
        \\min_{\\mathbf{w}, \\alpha \\ge 0} \\ \\alpha \\log \\mathbb{E}_{\\mathbb{P}} \\left[\\left(1 + f \\cdot \\mathbf{w}^\\top \\mathbf{r}\\right)^{-1/\\alpha}\\right] + \\alpha \\epsilon
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

        **Returns:**

        - `np.ndarray`: Vector of optimized portfolio weights.

        Raises:
            DataError: For any data mismatch during integrity check.
            PortfolioError: For any invalid portfolio variable inputs during integrity check.
            OptimizationError: If `SLSQP` solver fails to solve.

        !!! example "Example:"
            ```python
            # Importing the dro Kelly module
            from opes.objectives.distributionally_robust import KLRobustKelly

            # Let this be your ticker data
            training_data = some_data()

            # Initialize with custom fractional exposure and KL divergence radius
            kl_kelly = KLRobustKelly(fraction=0.8, radius=0.02)

            # Optimize portfolio with custom weight bounds
            weights = kl_kelly.optimize(data=training_data, weight_bounds=(0.05, 0.75))
            ```
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self._prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds, constraint_type=2)
        param_array = np.append(self.weights, 1)

        # Optimization objective and results
        def f(x):
            w, dual_var = x[:-1], x[-1]
            E = np.mean(
                np.maximum((1 + self.fraction * (trimmed_return_data @ w)), 0.001)
                ** (-1 / dual_var)
            )
            return dual_var * self.radius + dual_var * np.log(E)

        result = minimize(
            f,
            param_array,
            method="SLSQP",
            bounds=[weight_bounds] * len(self.weights) + [(1e-3, None)],
            constraints=constraint,
        )
        if result.success:
            self.weights = result.x[:-1]
            return self.weights
        else:
            raise OptimizationError(
                f"KL robust kelly criterion optimization failed: {result.message}"
            )


class WassRobustMaxMean(Optimizer):
    """
    Wasserstein Ambiguity Maximum Mean optimization.

    Maximum mean return under Wasserstein uncertainty has
    been studied extensively in the robust optimization literature. The
    Kantorovich-Rubinstein duality theorem provides an explicit dual reformulation.
    """

    def __init__(self, radius=0.01, ground_norm=2):
        """
        Args:
            radius (*float, optional*): The size of the uncertainty set (Wasserstein distance bound). Larger values indicate higher uncertainty. Defaults to `0.01`.
            ground_norm (*int, optional*): Wasserstein ground norm. Used to find the dual norm for the dual objective. Must be a positive integer. Defaults to `2`.
        """
        self.identity = "wassdr-mm"
        self.radius = radius
        self.ground_norm = ground_norm
        self.dual_norm = None
        self.mean = None

        self.tickers = None
        self.weights = None

    def _find_dual(self):
        if isinstance(self.ground_norm, str) and self.ground_norm.lower() == "inf":
            self.dual_norm = lambda w: np.sum(np.abs(w))
            return
        elif not isinstance(self.ground_norm, Real):
            raise PortfolioError(
                f"Expected ground_norm to be a number, got {self.ground_norm}"
            )
        elif self.ground_norm < 1:
            raise PortfolioError(f"ground_norm must ben >= 1, got {self.ground_norm}")
        elif self.ground_norm == 1:
            self.dual_norm = lambda w: np.max(np.abs(w))
            return
        else:
            dual_exponent = 1 / (1 - 1 / self.ground_norm)
            self.dual_norm = lambda w: np.sum(np.abs(w) ** dual_exponent) ** (
                1 / dual_exponent
            )
            return

    def _prepare_optimization_inputs(self, data, weight_bounds, w, custom_mean=None):
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        # Checking for mean and weights and assigning optimization data accordingly
        self.mean = np.mean(data, axis=0) if custom_mean is None else custom_mean
        self.tickers = extract_trim(data)[0]
        self.weights = np.array(
            np.ones(len(self.tickers)) / len(self.tickers) if w is None else w,
            dtype=float,
        )

        # Finding dual norm function
        self._find_dual()

        # Functions to test data integrity and find optimization constraint
        test_integrity(
            tickers=self.tickers,
            weights=self.weights,
            bounds=weight_bounds,
            uncertainty_radius=self.radius,
            mean=self.mean,
        )

    def optimize(self, data=None, weight_bounds=(0, 1), w=None, custom_mean=None):
        """
        Solves the Kantorovich-Rubinstein dual objective for type-1 Wasserstein distances:

        $$
        \\min_{\\mathbf{w}} \\ - \\mathbf{w}^\\top \\mu + \\epsilon \\| \\mathbf{w} \\|_{\\text{d}}
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
            # Importing the dro maximum mean module
            from opes.objectives.distributionally_robust import WassRobustMaxMean

            # Let this be your ticker data
            training_data = some_data()

            # Let this be your custom mean vector
            mean_v = customMean()

            # Initialize with ground norm and Wasserstein radius
            wass_maxmean = WassRobustMaxMean(radius=0.04, ground_norm=3)

            # Optimize portfolio with custom weight bounds and mean vector
            weights = wass_maxmean.optimize(data=training_data, weight_bounds=(0.05, 0.75), custom_mean=mean_v)
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
            return -w @ self.mean + self.radius * self.dual_norm(w)

        result = minimize(
            f,
            w,
            method="SLSQP",
            bounds=[weight_bounds] * len(self.weights),
            constraints=constraint,
        )
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(
                f"Wasserstein Robust Maximum Mean optimization failed: {result.message}"
            )


class WassRobustMinVariance(Optimizer):
    """
    Wasserstein Ambiguity Minimum Variance optimization.

    Builds on the distributionally robust
    optimization framework developed by Blanchet et al. The method extends the classical
    GMV portfolio by minimizing the worst-case portfolio variance over a Wasserstein
    ambiguity set centered at the empirical return distribution. Through duality, this
    worst-case problem admits a tractable reformulation that preserves convexity while
    explicitly controlling sensitivity to distributional misspecification. As a result,
    Wasserstein-robust GMV portfolios exhibit improved stability and out-of-sample
    performance relative to nominal GMV.
    """

    def __init__(self, radius=0.01, ground_norm=2):
        """
        Args:
            radius (*float, optional*): The size of the uncertainty set (Wasserstein distance bound). Larger values indicate higher uncertainty. Defaults to `0.01`.
            ground_norm (*int, optional*): Wasserstein ground norm. Used to find the dual norm for the dual objective. Must be a positive integer. Defaults to `2`.
        """
        self.identity = "wassdr-gmv"
        self.radius = radius
        self.ground_norm = ground_norm
        self.dual_norm = None
        self.covariance = None

        self.tickers = None
        self.weights = None

    def _find_dual(self):
        if isinstance(self.ground_norm, str) and self.ground_norm.lower() == "inf":
            self.dual_norm = lambda w: np.sum(np.abs(w))
            return
        elif not isinstance(self.ground_norm, Real):
            raise PortfolioError(
                f"Expected ground_norm to be a number, got {self.ground_norm}"
            )
        elif self.ground_norm < 1:
            raise PortfolioError(f"ground_norm must ben >= 1, got {self.ground_norm}")
        elif self.ground_norm == 1:
            self.dual_norm = lambda w: np.max(np.abs(w))
            return
        else:
            dual_exponent = 1 / (1 - 1 / self.ground_norm)
            self.dual_norm = lambda w: np.sum(np.abs(w) ** dual_exponent) ** (
                1 / dual_exponent
            )
            return

    def _prepare_optimization_inputs(self, data, weight_bounds, w, custom_cov=None):
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        # Checking for mean and weights and assigning optimization data accordingly
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
        self.tickers = extract_trim(data)[0]
        self.weights = np.array(
            np.ones(len(self.tickers)) / len(self.tickers) if w is None else w,
            dtype=float,
        )

        # Finding dual norm function
        self._find_dual()

        # Functions to test data integrity and find optimization constraint
        test_integrity(
            tickers=self.tickers,
            weights=self.weights,
            bounds=weight_bounds,
            uncertainty_radius=self.radius,
            cov=self.covariance,
        )

    def optimize(self, data=None, weight_bounds=(0, 1), w=None, custom_cov=None):
        """
        Solves the Wasserstein Ambiguity Minimum Variance dual objective:

        $$
        \\min_{\\mathbf{w}} \\ \\left(\\sqrt{\\mathbf{w}^\\top \\Sigma \\mathbf{w}} + \\sqrt{\\epsilon} \\| \\mathbf{w} \\|_{\\text{d}} \\right)^2
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
            # Importing the dro minimum variance module
            from opes.objectives.distributionally_robust import WassRobustMinVariance

            # Let this be your ticker data
            training_data = some_data()

            # Let this be your custom covariance matrix
            cov_m = customCov()

            # Initialize with ground norm and Wasserstein radius
            wass_minvariance = WassRobustMinVariance(radius=0.04, ground_norm=3)

            # Optimize portfolio with custom weight bounds and covariance matrix
            weights = wass_minvariance.optimize(data=training_data, weight_bounds=(0.05, 0.75), custom_cov=cov_m)
            ```
        """
        # Preparing optimization and finding constraint
        self._prepare_optimization_inputs(data, weight_bounds, w, custom_cov=custom_cov)
        constraint = find_constraint(weight_bounds)
        w = self.weights

        # Optimization objective and results
        def f(w):
            return (
                np.sqrt((w @ self.covariance @ w))
                + np.sqrt(self.radius) * self.dual_norm(w)
            ) ** 2

        result = minimize(
            f,
            w,
            method="SLSQP",
            bounds=[weight_bounds] * len(self.weights),
            constraints=constraint,
        )
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(
                f"Wasserstein Robust Minimum Variance optimization failed: {result.message}"
            )


class WassRobustMeanVariance(Optimizer):
    """
    Wasserstein Ambiguity Mean-Variance Optimization.

    Extends the classical mean-variance framework by incorporating distributional
    uncertainty using the Wasserstein distance, following the dual
    reformulation approach of Blanchet et al. Instead of optimizing
    expected return and variance under a single estimated distribution,
    the robust formulation considers the worst-case trade-off over a
    Wasserstein ambiguity set around the empirical data. The resulting
    dual problem remains tractable and convex, allowing robustness to
    be introduced without sacrificing computational feasibility. This
    approach mitigates sensitivity to estimation error in both the mean
    and covariance and yields portfolios with more reliable out-of-sample
    behavior under model misspecification.
    """

    def __init__(self, risk_aversion=0.3, radius=0.01, ground_norm=2):
        """
        Args:
            risk_aversion (*float, optional*): Risk-aversion coefficient. Higher values emphasize risk minimization, while lower values favor return seeking. Usually greater than `0`. Defaults to `0.3`.
            radius (*float, optional*): The size of the uncertainty set (Wasserstein distance bound). Larger values indicate higher uncertainty. Defaults to `0.01`.
            ground_norm (*int, optional*): Wasserstein ground norm. Used to find the dual norm for the dual objective. Must be a positive integer. Defaults to `2`.
        """
        self.identity = "wassdr-mvo"
        self.risk_aversion = risk_aversion
        self.radius = radius
        self.ground_norm = ground_norm
        self.dual_norm = None
        self.covariance = None
        self.mean = None

        self.tickers = None
        self.weights = None

    def _find_dual(self):
        if isinstance(self.ground_norm, str) and self.ground_norm.lower() == "inf":
            self.dual_norm = lambda w: np.sum(np.abs(w))
            return
        elif not isinstance(self.ground_norm, Real):
            raise PortfolioError(
                f"Expected ground_norm to be a number, got {self.ground_norm}"
            )
        elif self.ground_norm < 1:
            raise PortfolioError(f"ground_norm must ben >= 1, got {self.ground_norm}")
        elif self.ground_norm == 1:
            self.dual_norm = lambda w: np.max(np.abs(w))
            return
        else:
            dual_exponent = 1 / (1 - 1 / self.ground_norm)
            self.dual_norm = lambda w: np.sum(np.abs(w) ** dual_exponent) ** (
                1 / dual_exponent
            )
            return

    def _prepare_optimization_inputs(
        self, data, weight_bounds, w, custom_mean=None, custom_cov=None
    ):
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        # Checking for mean and weights and assigning optimization data accordingly
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
        self.tickers = extract_trim(data)[0]
        self.weights = np.array(
            np.ones(len(self.tickers)) / len(self.tickers) if w is None else w,
            dtype=float,
        )

        # Finding dual norm function
        self._find_dual()

        # Functions to test data integrity and find optimization constraint
        test_integrity(
            tickers=self.tickers,
            weights=self.weights,
            bounds=weight_bounds,
            uncertainty_radius=self.radius,
            mean=self.mean,
            cov=self.covariance,
        )

    def optimize(
        self, data=None, weight_bounds=(0, 1), w=None, custom_mean=None, custom_cov=None
    ):
        """
        Solves the Wasserstein Ambiguity Mean-Variance dual objective:

        $$
        \\min_{\\mathbf{w}} \\ \\frac\\lambda 2 \\left( \\sqrt{\\mathbf{w}^\\top \\Sigma \\mathbf{w}} + \\sqrt{\\epsilon} \\|\\mathbf{w} \\|_d \\right)^2 - \\mathbf{w}^\\top \\mu + \\epsilon \\| \\mathbf{w} \\|_d
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
            # Importing the dro mean-variance module
            from opes.objectives.distributionally_robust import WassRobustMeanVariance

            # Let this be your ticker data
            training_data = some_data()

            # Let these be your custom mean vector and covariance matrix
            mean_v = customMean()
            cov_m = customCov()

            # Initialize with risk aversion, ground norm and Wasserstein radius
            wass_mean_variance = WassRobustMeanVariance(risk_aversion=0.5, radius=0.04, ground_norm=3)

            # Optimize portfolio with custom weight bounds, mean vector and covariance matrix
            weights = wass_mean_variance.optimize(data=training_data, weight_bounds=(0.05, 0.75), custom_mean=mean_v, custom_cov=cov_m)
            ```
        """
        # Preparing optimization and finding constraint
        self._prepare_optimization_inputs(
            data, weight_bounds, w, custom_mean=custom_mean, custom_cov=custom_cov
        )
        constraint = find_constraint(weight_bounds)
        w = self.weights

        # Optimization objective and results
        def f(w):
            dual_value = self.dual_norm(w)
            base_mean = w @ self.mean
            base_variance = w @ self.covariance @ w

            return (
                (self.risk_aversion / 2)
                * (np.sqrt(base_variance) + np.sqrt(self.radius) * dual_value) ** 2
                - base_mean
                + self.radius * dual_value
            )

        result = minimize(
            f,
            w,
            method="SLSQP",
            bounds=[weight_bounds] * len(self.weights),
            constraints=constraint,
        )
        if result.success:
            self.weights = result.x
            return self.weights
        else:
            raise OptimizationError(
                f"Wasserstein Robust Mean-Variance Optimization failed: {result.message}"
            )
