from numbers import Integral as Integer, Real
import numpy as np
import pandas as pd

from opes.errors import DataError, PortfolioError


# Regularizer finding function
def find_regularizer(reg):
    """
    Return a regularization function based on the specified type.

    Parameters
    ----------
    reg : str or None
        The type of regularizer to use. Supported options are:
            - None: No regularization, returns a function that always outputs 0.
            - "l1": L1 regularization, returns the sum of absolute weights to encourage sparsity.
            - "l2": L2 regularization, returns the sum of squared weights.
            - "maxweight": Maximum weight regularization, returns the largest absolute weight.
            - "entropy": Entropy regularization, returns the sum of |w| * log(|w| + 1e-12) for numerical stability.
            - "variance": Variance regularization, returns the variance of the weights to encourage balanced allocations; returns 0 for a single weight.
            - "MPAD": Mean Absolute Pairwise Deviation regularization, returns the mean absolute pairwise difference between weights to encourage balance.

    Returns
    -------
    function
        A function that takes a weight vector `w` as input and computes the corresponding regularization value.

    Raises
    ------
    PortfolioError
        If `reg` is not one of the supported regularizer types.
    """
    regulizers = {
        None: lambda w: 0,
        "l1": lambda w: np.sum(np.abs(w)),
        "l2": lambda w: np.sum(w**2),
        "l-inf": lambda w: np.max(np.abs(w)),
        "entropy": lambda w: np.sum(np.abs(w) * np.log(np.abs(w) + 1e-12)),
        "variance": lambda w: np.var(w) if len(w) >= 2 else 0,
        "mpad": lambda w: np.mean(np.abs(w[:, None] - w[None, :])),
    }
    reg = str(reg).lower() if reg is not None else reg
    if reg in regulizers:
        return regulizers[reg]
    else:
        raise PortfolioError(f"Unknown regulizer: {reg}")


# Sequence element checker
def all_elements_are_type(sequence, target):
    """Check if all elements in a sequence are of the specified type."""
    return all(isinstance(i, target) for i in sequence)


# Extract and trim data for optimizers and backtesting engine. Returns tickers and returns
def extract_trim(data):
    """
    Extract adjusted close price returns from multi-level DataFrame and align series lengths.

    Parameters
    ----------
    data : pandas.DataFrame
        Multi-index DataFrame with tickers as the first level and OHLCV columns as the second level. Must contain 'Close' prices.
        Single-index DataFrame with tickers as the first level and per-day prices.

    Returns
    -------
    tickers : list
        List of unique ticker symbols from the DataFrame.
    returns : np.ndarray
        2D array of percentage returns for each ticker, trimmed to the shortest series length.

    Raises
    ------
    DataError
        If `data` is None.
        If 'Close' is not present within data
    """
    if data is None:
        raise DataError("Data not specified")
    # Check if columns have a MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        # If the columns have a MultiIndex then Close must be one of those indices
        if "Close" in data.columns.get_level_values(1):
            returnMatrix = (
                data.xs("Close", axis=1, level=1)
                .pct_change(fill_method=None)
                .dropna()
                .values.tolist()
            )
        else:
            raise DataError(
                "MultiIndex DataFrame detected, but level 1 does not contain a 'Close' column."
            )
    # If the column is single index, then the column is assumed to be close prices
    else:
        returnMatrix = data.pct_change(fill_method=None).dropna().values.tolist()
    # Obtaining tickers & truncating data to match column length without nans
    tickers = data.columns.get_level_values(0).unique().tolist()
    min_len = min(len(r) for r in returnMatrix)
    return tickers, np.array([r[-min_len:] for r in returnMatrix])


# Optimization constraints finding function
def find_constraint(bounds, constraint_type=1):
    """
    Generate linear equality constraints for portfolio optimization based on weight bounds.

    This function produces a list of constraint dictionaries compatible with SciPy optimizers.
    It handles both net-budget and gross-exposure constraints depending on the portfolio regime.

    Parameters
    ----------
    bounds : tuple of float
        Weight bounds in the format (min, max).
    constraint_type : int, optional
        Determines which weights to sum over:
            - 1: all weights
            - 2: all weights except the last one
        Default is 1.

    Returns
    -------
    list of dict
        Each dict represents a single equality constraint in the form required by SciPy's
        `minimize` function: {'type': 'eq', 'fun': callable}.

    Notes
    -----
    - If bounds span negative to positive (long-short), a **gross exposure** constraint
      (`sum(abs(weights)) = 1`) is added.
    - For fully long or fully short portfolios, a **net exposure** constraint
      (`sum(weights) = ±1`) is applied with a `shift` determined by bounds:
        - 0 for long-short
        - 1 for short-only
        - -1 for long-only
    - `constraint_type` controls whether the last weight is included in the sum,
      which can be useful for sequential optimization or pivoted variables.
    """
    constraint_list = []
    if bounds[0] < 0 and bounds[1] > 0:
        shift = 0
        # Setting Gross exposure to 1
        # Makes objective non-convex in general
        constraint_list.append({"type": "eq", "fun": lambda x: np.abs(x).sum() - 1})
    elif bounds[1] < 0:
        shift = 1
    else:
        shift = -1
    slicer = slice(None) if constraint_type == 1 else slice(None, -1)
    constraint_list.append({"type": "eq", "fun": lambda x: x[slicer].sum() + shift})
    return constraint_list


# Slippage function
def slippage(weights, returns, cost, numpy_seed=None):
    """
    Compute elementwise portfolio slippage given weights, returns, and cost model.

    Parameters
    ----------
    weights : np.ndarray, shape (T, N)
    returns : np.ndarray, shape (T, N)
    cost : dict
        Must have exactly one key. Supported models:
        - 'const': scalar
        - 'gamma': [shape, scale]
        - 'lognormal': [mean, sigma]
        - 'inversegaussian': [mean, scale]
        - 'jump': [lambda, mu, sigma] (compound Poisson)
    numpy_seed: int, numpy rng seed

    Returns
    -------
    turnover_array : np.ndarray, shape (T,)
    """
    numpy_rng = np.random.default_rng(numpy_seed)
    turnover_array = np.zeros(len(weights))
    # Loop range is from 1 to horizon. Rebalancing happens from t=1
    for i in range(1, len(weights)):
        w_current = weights[i]
        w_prev = weights[i - 1]
        w_realized = (w_prev * (1 + returns[i])) / (1 + np.sum(w_prev * returns[i]))
        turnover = np.sum(np.abs(w_current - w_realized))
        turnover_array[i] = turnover
    # Deciding slippage model using cost key
    cost_key = next(iter(cost)).lower()
    cost_params = cost[cost_key]
    # Constant slippage
    if cost_key == "const":
        return turnover_array * cost_params / 10000
    horizon = len(turnover_array)
    # Gamma distributed slippage
    if cost_key == "gamma":
        return (
            turnover_array
            * numpy_rng.gamma(shape=cost_params[0], scale=cost_params[1], size=horizon)
            / 10000
        )
    # Lognormally distributed slippage
    elif cost_key == "lognormal":
        return (
            turnover_array
            * numpy_rng.lognormal(
                mean=cost_params[0], sigma=cost_params[1], size=horizon
            )
            / 10000
        )
    # Inverse gaussian slippage
    elif cost_key == "inversegaussian":
        return (
            turnover_array
            * numpy_rng.wald(mean=cost_params[0], scale=cost_params[1], size=horizon)
            / 10000
        )
    # Compound poisson slippage (jump process)
    elif cost_key == "jump":
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
        return turnover_array * jump_cost / 10000
    raise DataError(f"Unknown cost model: {cost_key}")


# Data integrity checker
def test_integrity(
    tickers,
    weights=None,
    cov=None,
    mean=None,
    bounds=None,
    kelly_fraction=None,
    confidence=None,
    volatility_array=None,
    hist_bins=None,
    uncertainty_radius=None,
):
    """
    Validate the integrity and consistency of input portfolio data.

    Parameters
    ----------
    tickers : list
        List of asset tickers.
    weights : array-like, optional
        Portfolio weights; must match the number of tickers.
    cov : array-like, optional
        Covariance matrix; must be square with size equal to number of tickers and invertible.
    mean : array-like, optional
        Expected returns vector; length must match number of tickers.
    bounds : tuple of two reals, optional
        Weight bounds in the format (min, max) with min < max and abs(max), abs(min) <= 1.
    kelly_fraction : float, optional
        Fraction for Kelly criterion; must be in (0, 1].
    confidence : float, optional
        Confidence level for risk measures; must be in (0, 1).
    volatility_array : array-like, optional
        Asset volatilities; length must match number of tickers and all values > 0.
    hist_bins : int, optional
        Number of histogram bins; must be a positive integer.
    uncertainty_radius : float, optional
        Radius for uncertainty sets; must be positive.

    Raises
    ------
    DataError
        If any vector/matrix has incorrect type, length, or shape, or contains invalid values.
    PortfolioError
        If any portfolio-specific parameter (kelly_fraction, confidence, uncertainty_radius) is out of bounds.
    """
    asset_quantity = len(tickers)
    if mean is not None:
        if not all_elements_are_type(np.array(mean).flatten(), Real):
            raise DataError(f"Mean vector type mismatch. Expected real numbers")
        if len(mean) != asset_quantity:
            raise DataError(
                f"Mean vector shape mismatch. Expected {asset_quantity}, got {len(mean)}"
            )
    if cov is not None:
        if not all_elements_are_type(np.array(cov).flatten(), Real):
            raise DataError(f"Covariance Matrix type mismatch. Expected real numbers")
        if asset_quantity != cov.shape[0] or (cov.shape[0] != cov.shape[1]):
            raise DataError(
                f"Covariance matrix shape mismatch. Expected ({asset_quantity}, {asset_quantity}), got {cov.shape}"
            )
        try:
            np.linalg.inv(cov)
        except np.linal.LinAlgError:
            raise DataError(f"Singular covariance matrix")
    if weights is not None:
        if not all_elements_are_type(np.array(weights).flatten(), Real):
            raise DataError("Weights vector type mismatch. Expected real numbers")
        if len(weights) != asset_quantity:
            raise DataError(
                f"Weight vector shape mismatch. Expected {asset_quantity}, got {len(weights)}"
            )
    if bounds is not None:
        if not isinstance(bounds, tuple):
            raise DataError(
                f"Invalid bounds sequence type. Expected tuple, got {type(bounds)}"
            )
        if len(bounds) != 2:
            raise DataError(
                f"Invalid weight bounds length. Expected 2, got {len(bounds)}"
            )
        if not isinstance(bounds[0], Real) or not isinstance(bounds[1], Real):
            raise DataError(
                f"Invalid bounds type. Expected (real, real), got ({type(bounds[0])},{type(bounds[1])})"
            )
        if bounds[0] >= bounds[1]:
            raise DataError(
                f"Invalid weight bounds. Bounds must be of the format (start, end) with start < end"
            )
        if abs(bounds[1]) > 1 or abs(bounds[0]) > 1:
            raise DataError(
                f"Invalid weight bounds. Leverage not allowed, got ({bounds[0]}, {bounds[1]})"
            )
    if kelly_fraction is not None:
        if (
            not isinstance(kelly_fraction, Real)
            or kelly_fraction <= 0
            or kelly_fraction > 1
        ):
            raise PortfolioError(
                f"Invalid Kelly criterion fraction. Must be bounded within (0,1], got {kelly_fraction}"
            )
    if confidence is not None:
        if not isinstance(confidence, Real) or confidence <= 0 or confidence >= 1:
            raise PortfolioError(
                f"Invalid confidence value. Must be bounded within (0,1), got {confidence}"
            )
    if volatility_array is not None:
        if len(volatility_array) != asset_quantity:
            raise DataError(
                f"Volatility array length mismatch. Expected {len(tickers)}, got {len(volatility_array)}"
            )
        if (volatility_array <= 0).any():
            raise DataError(
                f"Invalid volatility values: volatility array must contain strictly positive values."
            )
    if hist_bins is not None:
        if not isinstance(hist_bins, Integer) or hist_bins <= 0:
            raise DataError(
                f"Invalid histogram bins. Expected integer within bounds [1, inf], got {hist_bins}"
            )
    if uncertainty_radius is not None:
        if not isinstance(uncertainty_radius, Real) or uncertainty_radius <= 0:
            raise PortfolioError(
                f"Invalid uncertainty set radius given. Expected real number within bounds (0, inf), got {uncertainty_radius}"
            )
