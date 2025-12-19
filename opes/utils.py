import time

import numpy as np
import scipy.stats as scistats
import matplotlib.pyplot as plt
import pandas as pd

from opes.errors import DataError, PortfolioError

def find_regularizer(reg):
    """
    Finds and returns the user specified portfolio regularizer.
    
    Parameters
    ----------
    reg: str
        Regularization scheme.

    Raises
    ------
    PortfolioError if regulizer is unknown.
    """
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

def test_integrity(tickers, weights=None, cov=None, mean=None, bounds=None):
    """
    Test data integrity before optimization.
    
    Parameters
    ----------
    tickers: sequence
        Used as a reference for length and shape to find discrepancies in other parameters
    weights: array-like
        Portfolio weights. Must have the same length of tickers
    cov: array-like
        Covariance matrix of asset returns. Must be square, match the length of `tickers`, and be invertible.
    mean: array-like
        mean vector of asset returns. Must match the length of `tickers`
    bounds: tuple[float, float]
        weight bounds for each asset. Must have a length of 2 and be in ascending order

    Raises
    ------
    DataError if any of the above conditions are not satisfied
    """
    if mean is not None:
        if len(mean) != len(tickers):
            raise DataError(f"Mean vector shape mismatch. Expected {len(tickers)}, Got {mean.shape}")
    if cov is not None:
        if len(tickers) != cov.shape[0] or (cov.shape[0] != cov.shape[1]):
            raise DataError(f"Covariance matrix shape mismatch. Expected ({len(tickers)}, {len(tickers)}), Got {cov.shape}")
        try:
            np.linalg.inv(cov)
        except np.linal.LinAlgError:
            raise DataError(f"Singular covariance matrix")
    if weights is not None:
        if len(weights) != len(tickers):
            raise DataError(f"Weight vector shape mismatch. Expected {len(tickers)}, Got {weights.shape}")
    if bounds is not None:
        bounds = tuple(bounds)
        if len(bounds) != 2:
            raise DataError(f"Invalid weight bounds length. Expected 2, Got {len(bounds)}")
        if bounds[0] >= bounds[1]:
            raise DataError(f"Invalid weight bounds. Bounds must be of the format (start, end)")

def extract_trim(data):
    """
    Extract and trim return data to enforce common trading history across all assets.
    
    Parameters
    ----------
    data: pandas.DataFrame
        OHLCV data indexed by time, with asset-specific columns.
    
    Returns
    -------
    array-like
        trimmed return data for all assets
    """
    if data is None:
        raise DataError("Data not specified")
    returnMatrix = data.xs('Close', axis=1, level=1).pct_change(fill_method=None).dropna().values.tolist()
    min_len = min(len(r) for r in returnMatrix)
    tickers = data.columns.get_level_values(0).unique().tolist()
    return tickers, np.array([r[-min_len:] for r in returnMatrix]).T

def find_constraint(bounds):
    """
    Returns a sum constraint function based on per-asset bounds.

    Parameters
    ----------
    bounds : tuple[float, float]
        Lower and upper bound for individual weights.

    Returns
    -------
    function
        Constraint function f(x) such that f(x) == 0 enforces
        the desired sum of weights.
    """
    if bounds[0] < 0 and bounds[1] > 0:
        return lambda x: x.sum()
    elif bounds[0] < 0 and bounds[1] < 0:
        return lambda x: x.sum() + 1
    else:
        return lambda x: x.sum() - 1

# Slippage function
def slippage(weights, previous_returns, cost):
    realized_weights = weights * (1 + previous_returns)
    realized_weights /= realized_weights.sum()
    turnover = np.sum(np.abs(weights - realized_weights))
    return cost * turnover

# Performance metrics analyzer
def metrics(returns, T):
    """
    Analyze strategy performance using multiple metrics.
    
    Parameters
    ----------
    returns: array-like
        Strategy returns to be analyzed
    T: int
        Back-test horizon

    Returns
    -------
    dict
        Dictionary containing performance metrics.
    """
    returns = np.array(returns)
    average = returns.mean()
    downside_vol = returns[returns < 0].std()
    vol = returns.std()

    # Performance metrics
    SHARPE = np.sqrt(252) * average / vol if (vol > 0 or not np.isnan(vol)) else np.nan
    SORTINO = np.sqrt(252) * average / downside_vol if (downside_vol > 0 or not np.isnan(downside_vol)) else np.nan
    VOLATILITY = vol * np.sqrt(252) if (vol > 0 or not np.isnan(vol)) else np.nan
    AVERAGE = average
    TOTAL = np.prod(1 + returns) - 1
    CAGR = (1 + TOTAL) ** (252/T) - 1
    MAX_DD = np.max(1 - np.cumprod(1 + returns) / np.maximum.accumulate(np.cumprod(1 + returns)))
    CALMAR = CAGR / abs(MAX_DD) if MAX_DD > 0 else np.nan
    VAR = -np.quantile(returns, 0.05)
    tail_returns = returns[returns <= -VAR]
    CVAR = -tail_returns.mean() if len(tail_returns) > 0 else np.nan
    SKEW = scistats.skew(returns)
    KURTOSIS = scistats.kurtosis(returns)
    
    # Zipping Text and values
    performance_metrics = [
        'Sharpe Ratio',
        'Sortino Ratio',
        'Calmar Ratio',
        'Volatility (%)',
        'Mean Return (%)',
        'Total Return (%)',
        'CAGR (%)',
        'Max Drawdown (%)',
        'VaR-95 (%)',
        'CVaR-95 (%)',
        'Skew',
        'Kurtosis'
    ]
    values = [VOLATILITY, AVERAGE, TOTAL, CAGR, MAX_DD, VAR, CVAR]
    results = [round(SHARPE, 2), round(SORTINO, 2), round(CALMAR, 2)] + [round(x*100, 2) for x in values] + [round(SKEW, 2), round(KURTOSIS, 2)]

    return dict(zip(performance_metrics, results))

# Plotting function
def plotter(portfolio, benchmark, dates, show, save):
    
    # Converting returns to wealth processes
    portfolio = np.cumprod(1 + np.array(portfolio))
    benchmark = np.cumprod(1 + np.array(benchmark))

    # Colors
    portfolio_color = "#2E7D32"    
    benchmark_color = "#1976D2"    
    green_fill = "#4CAF50"         
    red_fill = "#EF5350"           
    breakeven_color = "#000000"    

    # Plotting
    plt.plot(dates, portfolio, color=portfolio_color, label="Portfolio", linestyle='-')
    plt.plot(dates, benchmark, color=benchmark_color, label="Benchmark", linestyle='--')
    plt.axhline(y=1, color=breakeven_color, linestyle=':', label="Breakeven")
    plt.title("Wealth Performance")
    plt.xlabel("Time")
    plt.ylabel("Wealth")
    plt.grid(True)
    plt.legend()

    # Fill
    plt.fill_between(dates, 1, portfolio, where=(portfolio >= 1), color=green_fill, alpha=0.1)
    plt.fill_between(dates, 1, benchmark, where=(benchmark >= 1), color=green_fill, alpha=0.1)
    plt.fill_between(dates, 1, portfolio, where=(portfolio <= 1), color=red_fill, alpha=0.2)
    plt.fill_between(dates, 1, benchmark, where=(benchmark <= 1), color=red_fill, alpha=0.2)

    # Saving and showing if necessary
    if save:
        plt.savefig(f"plot_{int(time.time()*1000)}.png", dpi=300, bbox_inches='tight')
    if show:
        plt.show()