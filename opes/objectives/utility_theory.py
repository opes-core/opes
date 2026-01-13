"""
Utility theory forms the mathematical basis for rational decision-making under uncertainty in portfolio
management, originating with Daniel Bernoulli's resolution of the St. Petersburg paradox, where he
proposed that investors maximize expected utility rather than expected wealth. This idea was rigorously
formalized by von Neumann and Morgenstern's expected utility theorem, which established axioms for
rational choice, and later extended by Arrow and Debreu through general equilibrium theory, with
Arrow's work on risk aversion providing practical tools to model investor preferences via utility functions.

---
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opes.objectives.base_optimizer import Optimizer
from ..regularizer import _find_regularizer
from ..utils import extract_trim, test_integrity, find_constraint
from ..errors import OptimizationError, PortfolioError


class Kelly(Optimizer):
    """
    Kelly Criterion & Fractional Variants.

    The Kelly Criterion, introduced by John Larry Kelly Jr., maximizes the expected geometric growth rate of wealth and is equivalent
    to CRRA utility with risk aversion parameter $\\gamma = 1$ (log utility), yielding the unique strategy
    that maximizes long-run wealth almost surely under repeated betting. While it optimally balances risk and
    return and avoids ruin, full Kelly can suffer large short- to medium-term drawdowns, making fractional Kelly
    essential in practice, as scaling positions by a fraction (typically 0.25-0.5) reduces volatility quadratically
    while sacrificing growth only linearly, improving robustness to estimation error and drawdown tolerance.

    """

    def __init__(self, fraction=1.0, reg=None, strength=1):
        """
        Args:
            fraction (*float, optional*): The Kelly fractionional exposure. Must be bounded within (0,1]. Defaults to `1.0`.
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "kelly"
        self.reg = _find_regularizer(reg)
        self.strength = strength
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
            kelly_fraction=self.fraction,
        )
        return data

    def optimize(self, data=None, weight_bounds=(0, 1), w=None):
        """
        Solves the Kelly Criterion Objective:

        $$
        \\min_{\\mathbf{w}} \\ -\\mathbb{E} \\left[\\ln \\left(1 + f \\cdot \\mathbf{w}^\\top \\mathbf{r}\\right) \\right] + \\lambda R(\\mathbf{w})
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
            # Importing the kelly criterion module
            from opes.objectives.utility_theory import Kelly

            # Let this be your ticker data
            training_data = some_data()

            # Initialize Kelly Criterion with custom fractional market exposure and regularizer
            kellycriterion = Kelly(fraction=0.85, reg='entropy', strength=0.05)

            # Optimize for Kelly with custom weight bounds
            weights_kelly = kellycriterion.optimize(data=training_data, weight_bounds=(0.05, 0.8))
            ```
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self._prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights

        # Optimization objective and results
        def f(w):
            X = self.fraction * np.maximum((trimmed_return_data @ w), -0.99)
            return -np.mean(np.log(1 + X)) + self.strength * self.reg(w)

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
                f"Kelly criterion optimization failed: {result.message}"
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
            # Import the Kelly Criterion class
            from opes.objectives.utility_theory import Kelly

            # Set with 'entropy' regularization
            optimizer = Kelly(reg='entropy', strength=0.01)

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


class QuadraticUtility(Optimizer):
    """
    Quadratic Utility.

    Introduced by Harry Markowitz, quadratic utility provides the theoretical
    justification for mean-variance optimization, as it makes expected utility depend
    only on the mean and variance of returns. This leads to the elegant result that
    optimal portfolios lie on the efficient frontier.
    """

    def __init__(self, risk_aversion=0.5, reg=None, strength=1):
        """
        Args:
            risk_aversion (*float, optional*): Risk aversion for quadratic utility. Usually greater than `0`. Defaults to `0.5`.
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "quadutil"
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

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds)
        return data

    def optimize(self, data=None, weight_bounds=(0, 1), w=None):
        """
        Solves the Quadratic Utility objective:

        $$
        \\min_{\\mathbf{w}} \\ \\mathbb{E} \\left[ \\frac{\\gamma}{2}(1 + \\mathbf{w}^\\top \\mathbf{r})^2 - (1 + \\mathbf{w}^\\top \\mathbf{r}) \\right] + \\lambda R(\\mathbf{w})
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
            # Importing the Quadratic Utility class
            from opes.objectives.utility_theory import QuadraticUtility as QU

            # Let this be your ticker data
            training_data = some_data()

            # Initialize Quadratic Utility with custom risk aversion and regularizer
            qu_opt = QU(risk_aversion=0.2, reg='entropy', strength=0.05)

            # Optimize portfolio with custom weight bounds
            weights = qu_opt.optimize(data=training_data, weight_bounds=(0.05, 0.8))
            ```
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self._prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights

        # Optimization objective and results
        def f(w):
            X = 1 + np.maximum((trimmed_return_data @ w), -1)
            return np.mean(
                self.risk_aversion / 2 * (X**2) - X
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
                f"Quadratic utility optimization failed: {result.message}"
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
            # Import the Quadratic Utility class
            from opes.objectives.utility_theory import QuadraticUtility

            # Set with 'entropy' regularization
            optimizer = QuadraticUtility(reg='entropy', strength=0.01)

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


class CARA(Optimizer):
    """
    Constant Absolute Risk Aversion (CARA).

    Introduced by John W. Pratt in his analysis of risk aversion measures,
    CARA utility exhibits constant absolute risk aversion, meaning the investor's
    absolute risk tolerance (measured in currency) remains unchanged regardless of
    wealth level. This property makes CARA utility particularly tractable for problems
    with normally distributed returns, as optimal decisions become independent of
    initial wealth.
    """

    def __init__(self, risk_aversion=1, reg=None, strength=1):
        """
        Args:
            risk_aversion (*float, optional*): Risk aversion for CARA utility. Must be greater than `0`. Defaults to `1.0`.
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "cara"
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

        # Quick checking risk aversion validity
        if self.risk_aversion <= 0:
            raise PortfolioError(
                f"Invalid risk aversion. Expected within bounds (0, inf), got {self.risk_aversion}"
            )

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds)
        return data

    def optimize(self, data=None, weight_bounds=(0, 1), w=None):
        """
        Solves the CARA objective:

        $$
        \\min_{\\mathbf{w}} \\ \\mathbb{E} \\left[ \\frac{1}{\\alpha} e^{-\\alpha \\mathbf{w}^\\top \\mathbf{r}} \\right] + \\lambda R(\\mathbf{w})
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
            # Importing the CARA class
            from opes.objectives.utility_theory import CARA

            # Let this be your ticker data
            training_data = some_data()

            # Initialize CARA with custom risk aversion and regularizer
            cara_opt = CARA(risk_aversion=3, reg='entropy', strength=0.05)

            # Optimize portfolio with custom weight bounds
            weights = cara_opt.optimize(data=training_data, weight_bounds=(0.05, 0.8))
            ```
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self._prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights

        # Optimization objective and results
        def f(w):
            X = np.maximum((trimmed_return_data @ w), -1)
            return (1 / self.risk_aversion) * np.mean(
                np.exp(-self.risk_aversion * X)
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
            raise OptimizationError(f"CARA optimization failed: {result.message}")

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
            # Import the CARA class
            from opes.objectives.utility_theory import CARA

            # Set with 'entropy' regularization
            optimizer = CARA(reg='entropy', strength=0.01)

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


class CRRA(Optimizer):
    """
    Constant Relative Risk Aversion (CRRA).

    Introduced by Kenneth Arrow and John W. Pratt, CRRA utility, also known as power utility or
    isoelastic utility, maintains constant relative risk aversion. This means investors maintain
    constant portfolio proportions regardless of wealth level, a property consistent with
    empirical observations of investor behavior. $\\gamma$ represents both risk aversion and
    the inverse of the elasticity of intertemporal substitution.
    """

    def __init__(self, risk_aversion=2.0, reg=None, strength=1):
        """
        Args:
            risk_aversion (*float, optional*): Risk aversion for CRRA utility. Must be greater than `1`. Defaults to `2.0`.
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.
        """
        self.identity = "crra"
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

        # Checking for CRRA risk aversion validity
        if self.risk_aversion <= 1:
            raise PortfolioError(
                f"CRRA risk aversion out of bounds. Expected within (1,inf), Got {self.risk_aversion}"
            )

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds)
        return data

    def optimize(self, data=None, weight_bounds=(0, 1), w=None):
        """
        Solves the CRRA objective:

        $$
        \\min_{\\mathbf{w}} \\ - \\mathbb{E} \\left[ \\frac{1}{1-\\gamma} (1 + \\mathbf{w}^\\top \\mathbf{r})^{1-\\gamma} \\right] + \\lambda R(\\mathbf{w})
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
            # Importing the CRRA class
            from opes.objectives.utility_theory import CRRA

            # Let this be your ticker data
            training_data = some_data()

            # Initialize CRRA with custom risk aversion and regularizer
            crra_opt = CRRA(risk_aversion=4, reg='entropy', strength=0.05)

            # Optimize portfolio with custom weight bounds
            weights = crra_opt.optimize(data=training_data, weight_bounds=(0.05, 0.8))
            ```
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self._prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights

        # Optimization objective and results
        def f(w):
            X = np.maximum((trimmed_return_data @ w), -0.99)
            return -np.mean((1 + X) ** (1 - self.risk_aversion)) / (
                1 - self.risk_aversion
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
            raise OptimizationError(f"CRRA optimization failed: {result.message}")

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
            # Import the CRRA class
            from opes.objectives.utility_theory import CRRA

            # Set with 'entropy' regularization
            optimizer = CRRA(reg='entropy', strength=0.01)

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


class HARA(Optimizer):
    """
    Optimizer based on Hyperbolic Absolute Risk Aversion (HARA).

    HARA utility, which was developed by various researchers in the 1970s, with significant
    contributions from Robert C. Merton, is a general class that nests many common utility functions
    (CRRA, CARA, Quadratic Utility) as special cases. The absolute risk aversion is hyperbolic.
    HARA utility is particularly valuable in continuous-time portfolio problems and aggregation
    theorems, as it preserves tractability while allowing flexible risk preferences. The parameters
    can be calibrated to match observed portfolio behavior across different wealth levels.
    """

    def __init__(self, risk_aversion=2.0, scale=1.0, shift=3.0, reg=None, strength=1):
        """
        Initializes the HARA optimizer.
        Args:
            risk_aversion (*float, optional*): Risk aversion for HARA utility. Must be greater than `1.0`. Defaults to `2.0`.
            scale (*float, optional*): Scaling factor for the wealth term. Must be greater than `0`. Defaults to `1.0`.
            shift (*float, optional*): Shift parameter for the utility function. Defaults to `3.0`.
            reg (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
            strength (*float, optional*): Strength of the regularization. Defaults to `1`.

        !!! note "Note"
            Please do not approximate CRRA, CARA or Quadratic Utility using HARA. Each utility has its own objective to preserve
            interpretability and numerical behavior.
        """
        self.identity = "hara"
        self.reg = _find_regularizer(reg)
        self.strength = strength
        self.risk_aversion = risk_aversion
        self.scale = scale
        self.shift = shift

        self.tickers = None
        self.weights = None

    def _prepare_optimization_inputs(self, data, weight_bounds, w):
        # Extracting trimmed return data from OHLCV and obtaining tickers and Checking for initial weights
        self.tickers, data = extract_trim(data)
        self.weights = np.array(
            np.ones(len(self.tickers)) / len(self.tickers) if w is None else w,
            dtype=float,
        )

        # Checking for CRRA risk aversion validity
        if self.risk_aversion <= 1:
            raise PortfolioError(
                f"HARA risk aversion out of bounds. Expected within (1,inf), Got {self.risk_aversion}"
            )
        if self.scale <= 0:
            raise PortfolioError(
                f"HARA scale value out of bounds. Expected within (0, inf), Got {self.scale}"
            )

        # Functions to test data integrity and find optimization constraint
        test_integrity(tickers=self.tickers, weights=self.weights, bounds=weight_bounds)
        return data

    def optimize(self, data=None, weight_bounds=(0, 1), w=None):
        """
        Solves the HARA objective:

        $$
        \\min_{\\mathbf{w}} \\ - \\mathbb{E} \\left[ \\frac{1-\\gamma}{\\gamma} \\left(\\frac{a(\\mathbf{w}^\\top \\mathbf{r})}{1-\\gamma} + b \\right)^{\\gamma} \\right] + \\lambda R(\\mathbf{w})
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
            # Importing the HARA class
            from opes.objectives.utility_theory import HARA

            # Let this be your ticker data
            training_data = some_data()

            # Initialize HARA optimizer with custom risk aversion, shape, scale and regularizer
            hara_opt = HARA(risk_aversion=3, shape=1.3, scale=0.5, reg='entropy', strength=0.05)

            # Optimize portfolio with custom weight bounds
            weights = hara_opt.optimize(data=training_data, weight_bounds=(0.05, 0.8))
            ```
        """
        # Preparing optimization and finding constraint
        trimmed_return_data = self._prepare_optimization_inputs(data, weight_bounds, w)
        constraint = find_constraint(weight_bounds)
        w = self.weights

        # Optimization objective and results
        def f(w):
            X = np.maximum((trimmed_return_data @ w), -0.99)
            return (self.risk_aversion - 1) * np.mean(
                (self.scale * (1 + X) / (1 - self.risk_aversion) + self.shift)
                ** (self.risk_aversion)
            ) / (self.risk_aversion) + self.strength * self.reg(w)

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
            raise OptimizationError(f"HARA optimization failed: {result.message}")

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
            # Import the HARA class
            from opes.objectives.utility_theory import HARA

            # Set with 'entropy' regularization
            optimizer = HARA(reg='entropy', strength=0.01)

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
