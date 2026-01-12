"""
Traditional portfolio optimization, especially the Markowitz framework,
uses variance as the primary risk measure, but this approach is fundamentally
limited because it penalizes upside and downside deviations equally, treats
tail events symmetrically and fails to capture catastrophic downside risk.
Modern alternatives instead focus on tail risk through coherent risk measures,
as formalized by Artzner et al., which satisfy properties such as subadditivity,
monotonicity, positive homogeneity and translation invariance, providing a
principled way to measure downside risk. This shift reflects a more realistic
view of markets and investor preferences, recognizing asymmetric losses, fat
tails, skewness, and tail dependence that dominate portfolio behavior during
crises and motivate more robust portfolio construction.

!!! tip "Author's Recommendation"
    If you plan on using these portfolios and have low risk assets like ETFs or
    cash proxies, consider using a regularizer. Risk measures are very pessimistic
    and can allocate majority of the capital towards these assets, making them
    extremely conservative (Yes, EVaR, I'm looking at you).

---
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opes.objectives.base_optimizer import Optimizer
from ..regularizer import _find_regularizer
from ..utils import extract_trim, test_integrity, find_constraint
from ..errors import OptimizationError, PortfolioError


class CVaR(Optimizer):
    """
    Conditional-Value-at-Risk optimization.

    Conditional Value-at-Risk (CVaR), also known as Expected Shortfall,
    was introduced by Rockafellar and Uryasev as a coherent alternative
    to Value-at-Risk by measuring the expected loss conditional on
    losses exceeding the VaR threshold. Unlike VaR, which only identifies
    a cutoff, CVaR captures the full severity of tail losses, is convex
    and coherent and therefore rewards diversification while enabling
    efficient optimization. The confidence level (commonly 0.95 or 0.99)
    determines how extreme the measured tail is, with higher values
    focusing on rarer events, and these properties have made CVaR a
    standard risk measure in banking regulation, insurance and
    institutional portfolio management.
    """

    def __init__(self, confidence=0.95, reg=None, strength=0):
        """
        Args:
            confidence (*float, optional*): The confidence level for tail calculation. Must be bounded within (0,1). Defaults to `0.95`.
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "cvar"
        self.reg = _find_regularizer(reg)
        self.strength = strength
        self.alpha = confidence

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
            confidence=self.alpha,
        )
        return data

    def optimize(self, data=None, weight_bounds=(0, 1), w=None):
        """
        Solves the Rockafellar-Uryasev Linear Programming objective:

        $$
        \\min_{\\mathbf{w}, \\zeta} \\ \\zeta + \\frac{1}{1-\\alpha} \\mathbb{E} \\left[\\left(-\\mathbf{w}^\\top \\mathbf{r} - \\zeta \\right)_+\\right] + \\lambda R(\\mathbf{w})
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
            # Importing the CVaR module
            from opes.objectives.risk_measures import CVaR

            # Let this be your ticker data
            training_data = some_data()

            # Initialize with confidence value and custom regularization
            cvarportfolio = CVaR(confidence=0.90, reg='entropy', strength=0.02)

            # Optimize portfolio with custom weight bounds
            weights = cvarportfolio.optimize(data=training_data, weight_bounds=(0.05, 0.75))
            ```
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self._prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds, constraint_type=2)
        w = self.weights

        # Optimization objective and results
        # Appending initial VaR value, 1, to parameter array
        param_array = np.append(w, 1)

        def f(x):
            w, v = x[:-1], x[-1]
            X = -trimmed_return_data @ w
            excess = np.mean(np.maximum(X - v, 0.0))
            return v + excess / (1 - self.alpha) + self.strength * self.reg(w)

        result = minimize(
            f,
            param_array,
            method="SLSQP",
            bounds=[weight_bounds] * len(w) + [(None, None)],
            constraints=constraint,
        )
        if result.success:
            self.weights = result.x[:-1]
            return self.weights
        else:
            raise OptimizationError(f"CVaR optimization failed: {result.message}")

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
            # Import the CVaR class
            from opes.objectives.risk_measures import CVaR

            # Set with 'entropy' regularization
            optimizer = CVaR(reg='entropy', strength=0.01)

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


class MeanCVaR(Optimizer):
    """
    Mean-CVaR optimization.

    Mean-CVaR optimization, introduced by Rockafellar and Uryasev, extends mean-variance optimization by
    trading off expected return against Conditional Value-at-Risk, producing an efficient frontier in
    mean-CVaR space that explicitly accounts for tail risk. This framework reflects the asymmetry of
    investor preferences by prioritizing the control of large losses while still rewarding expected
    gains, avoiding the tendency of mean-variance methods to concentrate in fat-tailed assets. As a
    result, mean-CVaR has become a standard approach in settings where tail risk matters most, such as
    hedge funds, pension funds and other risk-sensitive institutional portfolios.
    """

    def __init__(self, risk_aversion=0.5, confidence=0.95, reg=None, strength=0):
        """
        Args:
            risk_aversion (*float, optional*): Weight applied to the CVaR component. Usually greater than `0`. Defaults to `0.5`.
            confidence (*float, optional*): The confidence level for tail calculation. Must be bounded within (0,1). Defaults to `0.95`.
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "mcvar"
        self.reg = _find_regularizer(reg)
        self.strength = strength
        self.alpha = confidence
        self.risk_aversion = risk_aversion
        self.mean = None

        self.tickers = None
        self.weights = None

    def _prepare_optimization_inputs(self, data, weight_bounds, w, custom_mean=None):
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
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
            confidence=self.alpha,
        )
        return data

    def optimize(self, data=None, weight_bounds=(0, 1), w=None, custom_mean=None):
        """
        Solves the Mean-CVaR objective:

        $$
        \\min_{\\mathbf{w}} \\ \\gamma \\text{CVaR}_{\\alpha}(\\mathbf{w}^\\top \\mathbf{r}) - \\mathbf{w}^\\top \\mu + \\lambda R(\\mathbf{w})
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
            # Importing the Mean-CVaR module
            from opes.objectives.risk_measures import MeanCVaR

            # Let this be your ticker data
            training_data = some_data()

            # Let this be your custom mean vector
            # Can be Black-Litterman, Bayesian, Fama-French etc.
            mean_v = customMean()

            # Initialize with risk_aversion, confidence and custom regularization
            meancvarportfolio = MeanCVaR(risk_aversion=0.9, confidence=0.9, reg='entropy', strength=0.02)

            # Optimize portfolio with custom weight bounds and custom mean vector
            weights = meancvarportfolio.optimize(data=training_data, weight_bounds=(0.05, 0.75), custom_mean=mean_v)
            ```
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self._prepare_optimization_inputs(
            data, weight_bounds, w, custom_mean=custom_mean
        )
        constraint = find_constraint(weight_bounds, constraint_type=2)
        w = self.weights

        # Optimization objective and results
        # Appending initial VaR value, 1, to parameter array
        param_array = np.append(w, 1)

        def f(x):
            w, v = x[:-1], x[-1]
            X = -trimmed_return_data @ w
            excess = np.mean(np.maximum(X - v, 0.0))
            mean = self.mean @ w
            return (
                self.risk_aversion * (v + excess / (1 - self.alpha))
                + self.strength * self.reg(w)
                - mean
            )

        result = minimize(
            f,
            param_array,
            method="SLSQP",
            bounds=[weight_bounds] * len(w) + [(None, None)],
            constraints=constraint,
        )
        if result.success:
            self.weights = result.x[:-1]
            return self.weights
        else:
            raise OptimizationError(f"Mean CVaR optimization failed: {result.message}")

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
            # Import the MeanCVaR class
            from opes.objectives.risk_measures import MeanCVaR

            # Set with 'entropy' regularization
            optimizer = MeanCVaR(reg='entropy', strength=0.01)

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


class EVaR(Optimizer):
    """
    Entropic-Value-at-Risk optimization.

    Entropic Value-at-Risk (EVaR), introduced by Ahmadi-Javid, is a coherent risk measure
    grounded in exponential utility and relative entropy, arising as the tightest upper bound
    on CVaR over distributions within a fixed relative-entropy neighborhood of a reference
    distribution. Parameterized by a risk variable (not to be confused with standard risk-aversion),
    EVaR smoothly interpolates between expected loss in the low-risk limit and worst-case loss in the high-risk limit,
    making it strictly more sensitive to tail risk than CVaR. This combination of information-theoretic
    foundations and strong tail sensitivity makes EVaR suitable for highly risk-averse settings and
    conservative regulatory applications.
    """

    def __init__(self, confidence=0.85, reg=None, strength=0):
        """
        Args:
            confidence (*float, optional*): The confidence level for tail calculation. Must be bounded within (0,1). Defaults to `0.85`.
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "evar"
        self.reg = _find_regularizer(reg)
        self.strength = strength
        self.alpha = confidence

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
            confidence=self.alpha,
        )
        return data

    def optimize(self, data=None, weight_bounds=(0, 1), w=None):
        """
        Solves the EVaR objective:

        $$
        \\min_{\\mathbf{w}, s > 0} \\ \\frac{1}{s} \\ln \\left( \\frac{\\mathbb{E}\\left[ e^{-s \\left(\\mathbf{w}^\\top \\mathbf{r}\\right)}\\right]}{1-\\alpha} \\right) + \\lambda R(\\mathbf{w})
        $$

        !!! note "Note"
            In OPES, EVaR's risk-aversion parameter, $s$, is not fixed a priori, but is optimized jointly with the portfolio's loss distribution.
            This removes arbitrary tuning and yields a coherent, scale-consistent portfolio that is less susceptible to extreme losses and
            drawdowns for a given confidence level.

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
            # Importing the EVaR module
            from opes.objectives.risk_measures import EVaR

            # Let this be your ticker data
            training_data = some_data()

            # Initialize with confidence value and custom regularization
            evarportfolio = EVaR(confidence=0.90, reg='entropy', strength=0.02)

            # Optimize portfolio with custom weight bounds
            weights = evarportfolio.optimize(data=training_data, weight_bounds=(0.05, 0.75))
            ```
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self._prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds, constraint_type=2)
        w = self.weights

        # Optimization objective and results
        # Appending risk variable value as 1 to parameter array
        param_array = np.append(w, 1)

        def f(x):
            w, s = x[:-1], x[-1]
            X = trimmed_return_data @ w
            return (1 / s) * (
                np.log(np.mean(np.exp(-s * X))) - np.log(1 - self.alpha)
            ) + self.strength * self.reg(w)

        result = minimize(
            f,
            param_array,
            method="SLSQP",
            bounds=[weight_bounds] * len(w) + [(1e-8, None)],
            constraints=constraint,
        )
        if result.success:
            self.weights = result.x[:-1]
            return self.weights
        else:
            raise OptimizationError(f"EVaR optimization failed: {result.message}")

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
            # Import the EVaR class
            from opes.objectives.risk_measures import EVaR

            # Set with 'entropy' regularization
            optimizer = EVaR(reg='entropy', strength=0.01)

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


class MeanEVaR(Optimizer):
    """
    Mean-EVaR optimization.

    Mean-EVaR optimization extends the mean-CVaR framework by replacing
    CVaR with the more tail-sensitive Entropic Value-at-Risk, balancing
    expected return against entropic tail risk while remaining a convex
    and computationally tractable optimization problem. This framework
    is especially suited to settings with significant tail uncertainty
    or conservative regulatory constraints, as EVaR's exponential
    weighting of extreme losses discourages positions that may appear
    attractive under mean-CVaR but carry severe downside risk, allowing
    finer control over portfolio conservatism.
    """

    def __init__(self, risk_aversion=0.5, confidence=0.85, reg=None, strength=0):
        """
        Args:
            risk_aversion (*float, optional*): Weight applied to the EVaR component. Usually greater than `0`. Defaults to `0.5`.
            confidence (*float, optional*): The confidence level for tail calculation. Must be bounded within (0,1). Defaults to `0.85`.
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "mevar"
        self.reg = _find_regularizer(reg)
        self.strength = strength
        self.alpha = confidence
        self.risk_aversion = risk_aversion
        self.mean = None

        self.tickers = None
        self.weights = None

    def _prepare_optimization_inputs(self, data, weight_bounds, w, custom_mean=None):
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.mean = np.mean(data, axis=0) if custom_mean is None else custom_mean
        self.weights = np.array(
            np.ones(len(self.tickers)) / len(self.tickers) if w is None else w,
            dtype=float,
        )

        # Functions to test data integrity and find optimization constraint
        test_integrity(
            tickers=self.tickers,
            weights=self.weights,
            bounds=weight_bounds,
            confidence=self.alpha,
        )
        return data

    def optimize(self, data=None, weight_bounds=(0, 1), w=None, custom_mean=None):
        """
        Solves the Mean-EVaR objective:

        $$
        \\min_{\\mathbf{w}} \\ \\gamma \\text{EVaR}_{\\alpha}(\\mathbf{w}^\\top \\mathbf{r}) - \\mathbf{w}^\\top \\mu + \\lambda R(\\mathbf{w})
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
            # Importing the Mean-EVaR module
            from opes.objectives.risk_measures import MeanEVaR

            # Let this be your ticker data
            training_data = some_data()

            # Let this be your custom mean vector
            # Can be Black-Litterman, Bayesian, Fama-French etc.
            mean_v = customMean()

            # Initialize with risk_aversion, confidence and custom regularization
            meanevarportfolio = MeanEVaR(risk_aversion=0.9, confidence=0.9, reg='entropy', strength=0.02)

            # Optimize portfolio with custom weight bounds and custom mean vector
            weights = meanevarportfolio.optimize(data=training_data, weight_bounds=(0.05, 0.75), custom_mean=mean_v)
            ```
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self._prepare_optimization_inputs(
            data, weight_bounds, w, custom_mean=custom_mean
        )
        constraint = find_constraint(weight_bounds, constraint_type=2)
        w = self.weights

        # Optimization objective and results
        # Appending dual variable value as 1 to parameter array
        param_array = np.append(w, 1)

        def f(x):
            w, s = x[:-1], x[-1]
            X = trimmed_return_data @ w
            mean = self.mean @ w
            return (
                self.risk_aversion
                * ((1 / s) * (np.log(np.mean(np.exp(-s * X))) - np.log(1 - self.alpha)))
                + self.strength * self.reg(w)
                - mean
            )

        result = minimize(
            f,
            param_array,
            method="SLSQP",
            bounds=[weight_bounds] * len(w) + [(1e-8, None)],
            constraints=constraint,
        )
        if result.success:
            self.weights = result.x[:-1]
            return self.weights
        else:
            raise OptimizationError(f"Mean EVaR optimization failed: {result.message}")

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
            # Import the MeanEVaR class
            from opes.objectives.risk_measures import MeanEVaR

            # Set with 'entropy' regularization
            optimizer = MeanEVaR(reg='entropy', strength=0.01)

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


class EntropicRisk(Optimizer):
    """
    Optimizer for minimizing the Entropic Risk Measure (ERM).

    The Entropic Risk Measure (ERM), introduced by Föllmer and Schied, is
    a convex risk measure derived from exponential utility and defined as
    the negative exponential certainty equivalent, representing the
    guaranteed amount an investor with given risk aversion would accept
    in place of a random payoff. ERM admits an information-theoretic
    interpretation as a worst-case expected loss over probability measures
    constrained by relative entropy, naturally capturing model uncertainty
    and distributional ambiguity. The risk-aversion parameter controls
    sensitivity, interpolating smoothly between expected loss for small values
    and worst-case loss as risk aversion increases, making ERM well suited for
    robust portfolio optimization.

    !!! note "Note"
        ERM measures risk, so lower values indicate less risk. Minimizing ERM is equivalent to maximizing exponential utility. However,
        when regularization is enabled, the objectives may solve for distinct weights for a given `strength` value.

    """

    def __init__(self, risk_aversion=1, reg=None, strength=1):
        """
        Args:
            risk_aversion (*float, optional*): Weight applied to the EVaR component. Must be greater than `0`. Defaults to `1`.
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "erm"
        self.reg = _find_regularizer(reg)
        self.strength = strength
        self.risk_aversion = risk_aversion

        self.tickers = None
        self.weights = None

    def _prepare_optimization_inputs(self, data, weight_bounds, w):
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(
            np.ones(len(self.tickers)) / len(self.tickers) if w is None else w,
            dtype=float,
        )

        # Checking ERM risk aversion bounds
        if self.risk_aversion == 0:
            raise PortfolioError(
                f"Invalid ERM risk aversion. Expected within bounds (0, inf), Got {self.risk_aversion}"
            )

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds)
        return data

    def optimize(self, data=None, weight_bounds=(0, 1), w=None):
        """
        Solves the Entropic Risk Measure objective:

        $$
        \\min_{\\mathbf{w}} \\ \\frac{1}{\\theta} \\log \\left( \\mathbb{E}\\left[ e^{-\\theta \\mathbf{w}^\\top \\mathbf{r}}\\right] \\right) + \\lambda R(\\mathbf{w})
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
            # Importing the ERM module
            from opes.objectives.risk_measures import EntropicRisk

            # Let this be your ticker data
            training_data = some_data()

            # Initialize with confidence value and custom regularization
            erm_portfolio = EntropicRisk(confidence=0.90, reg='entropy', strength=0.02)

            # Optimize portfolio with custom weight bounds
            weights = erm_portfolio.optimize(data=training_data, weight_bounds=(0.05, 0.75))
            ```
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self._prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights

        # Optimization objective and results
        def f(w):
            X = trimmed_return_data @ w
            return 1 / self.risk_aversion * np.log(
                np.mean(np.exp(-self.risk_aversion * X))
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
            return self.weights
        else:
            raise OptimizationError(
                f"Entropic risk metric optimization failed: {result.message}"
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
            # Import the EntropicRisk class
            from opes.objectives.risk_measures import EntropicRisk

            # Set with 'entropy' regularization
            optimizer = EntropicRisk(reg='entropy', strength=0.01)

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
