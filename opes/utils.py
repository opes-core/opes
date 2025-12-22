import numpy as np
import pandas as pd

from opes.errors import DataError, PortfolioError

def find_regularizer(reg):
    regulizers = {
        None: lambda w: 0,
        "l2": lambda w: np.sum(w ** 2),
        "maxweight": lambda w: max(np.abs(w)),
        "entropy": lambda w: np.sum(np.abs(w) * np.log(np.abs(w) + 1e-12)),
    }
    reg = str(reg).lower() if reg is not None else reg
    if reg in regulizers:
        return regulizers[reg]
    else:
        raise PortfolioError(f"Unknown regulizer: {reg}")

def test_integrity(
        tickers, 
        weights=None, 
        cov=None, 
        mean=None,
        bounds=None, 
        kelly_fraction=None, 
        confidence=None, 
        volatility_array=None
    ):
    asset_quantity = len(tickers)
    if mean is not None:
        if len(mean) != asset_quantity:
            raise DataError(f"Mean vector shape mismatch. Expected {asset_quantity}, Got {len(mean)}")
    if cov is not None:
        if asset_quantity != cov.shape[0] or (cov.shape[0] != cov.shape[1]):
            raise DataError(f"Covariance matrix shape mismatch. Expected ({asset_quantity}, {asset_quantity}), Got {cov.shape}")
        try:
            np.linalg.inv(cov)
        except np.linal.LinAlgError:
            raise DataError(f"Singular covariance matrix")
    if weights is not None:
        if len(weights) != asset_quantity:
            raise DataError(f"Weight vector shape mismatch. Expected {asset_quantity}, Got {len(weights)}")
    if bounds is not None:
        bounds = tuple(bounds)
        if len(bounds) != 2:
            raise DataError(f"Invalid weight bounds length. Expected 2, Got {len(bounds)}")
        if bounds[0] >= bounds[1]:
            raise DataError(f"Invalid weight bounds. Bounds must be of the format (start, end)")
    if kelly_fraction is not None:
        if kelly_fraction <= 0 or kelly_fraction > 1:
            raise DataError(f"Invalid Kelly criterion fraction. Must be bounded within (0,1], Got {kelly_fraction}")
    if confidence is not None:
        if confidence <=0 or confidence >= 1:
            raise DataError(f"Invalid confidence value. Must be bounded within (0,1), Got {confidence}")
    if volatility_array is not None:
        if len(volatility_array) != asset_quantity:
            raise DataError(f"Volatility array length mismatch. Expected {len(tickers)}, Got {len(volatility_array)}")
        if (volatility_array <= 0).any():
            raise DataError(f"Invalid volatility values: volatility array must contain strictly positive values.")

def extract_trim(data):
    if data is None:
        raise DataError("Data not specified")
    returnMatrix = data.xs('Close', axis=1, level=1).pct_change(fill_method=None).dropna().values.tolist()
    min_len = min(len(r) for r in returnMatrix)
    tickers = data.columns.get_level_values(0).unique().tolist()
    return tickers, np.array([r[-min_len:] for r in returnMatrix]).T

def extract_data(data):
    if data is None:
        raise DataError("Data not specified")
    returnMatrix = data.xs('Close', axis=1, level=1).pct_change(fill_method=None).dropna().values.tolist()
    min_len = min(len(r) for r in returnMatrix)
    return np.array([r[-min_len:] for r in returnMatrix])

def find_constraint(bounds, constraint_type=1):
    if bounds[0] < 0 and bounds[1] > 0:
        shift = 0
    elif bounds[1] < 0:
        shift = 1
    else:
        shift = -1
    slicer = slice(None) if constraint_type == 1 else slice(None, -1)
    return lambda x: x[slicer].sum() + shift

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
    slippage_array : np.ndarray, shape (T,)
    """
    numpy_rng = np.random.default_rng(numpy_seed)
    slippage_array = np.zeros(len(weights))
    # Loop range is from 1 to horizon. Rebalancing happens from t=1
    for i in range(1, len(weights)):
        w_current = weights[i]
        w_prev = weights[i-1]
        w_realized = w_prev * (1 + returns[i])
        w_realized /= w_realized.sum()
        turnover = np.sum(np.abs(w_current - w_realized))
        slippage_array[i] = turnover
    # Deciding slippage model using cost key
    cost_key = next(iter(cost)).lower()
    cost_params = cost[cost_key]
    # Constant slippage
    if cost_key == 'const':
        return slippage_array * cost_params / 10000
    horizon = len(slippage_array)
    # Gamma distributed slippage
    if cost_key == 'gamma':
        return slippage_array * numpy_rng.gamma(shape=cost_params[0], scale=cost_params[1], size=horizon) / 10000
    # Lognormally distributed slippage
    elif cost_key == 'lognormal':
        return slippage_array * numpy_rng.lognormal(mean=cost_params[0], sigma=cost_params[1], size=horizon) / 10000
    # Inverse gaussian slippage
    elif cost_key == 'inversegaussian':
        return slippage_array * numpy_rng.wald(mean=cost_params[0], scale=cost_params[1], size=horizon) / 10000
    # Compound poisson slippage (jump process)
    elif cost_key == 'jump':
        N = numpy_rng.poisson(cost_params[0], size=horizon)
        jump_cost = np.array([np.sum(numpy_rng.lognormal(mean=cost_params[1], sigma=cost_params[2], size=n)) if n > 0 else 0 for n in N])
        return slippage_array * jump_cost / 10000
    raise DataError(f"Unknown cost model: {cost_key}")