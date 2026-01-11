from numbers import Integral as Integer, Real
import numpy as np
import pandas as pd

from opes.errors import DataError, PortfolioError


# Regularizer finding function
def find_regularizer(reg):
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
    constraint_list = []
    if bounds[1] < 0:
        shift = 1
    else:
        shift = -1
    slicer = slice(None) if constraint_type == 1 else slice(None, -1)
    constraint_list.append({"type": "eq", "fun": lambda x: x[slicer].sum() + shift})
    return constraint_list


# Slippage function
def slippage(weights, returns, cost, numpy_seed=None):
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
