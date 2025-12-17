# Standard lib
import time

# Third-party libs
import numpy as np
import scipy.stats as scistats
import matplotlib.pyplot as plt
import pandas as pd

# Local libs
from opes.errors import DataError

# Constant set
REQUIRED_FIELDS = {"Open", "High", "Low", "Close", "Volume"}

# Slippage function
def slippage(weights, previous_returns, cost):
    realized_weights = weights * (1 + previous_returns)
    realized_weights /= realized_weights.sum()
    turnover = np.sum(np.abs(weights - realized_weights))
    return cost * turnover

# Performance metrics analyzer
def metrics(returns, T):

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

# Data trimming function
def trimmer(tickers, data):
    returnMatrix = []
    for ticker in tickers:
        asset = data[ticker]["Close"].pct_change(fill_method=None).dropna().values
        returnMatrix.append(asset)
    min_len = min(len(r) for r in returnMatrix)
    return np.array([r[-min_len:] for r in returnMatrix]).T

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

# CSV reader function
def readCSV(path):
    try:
        data = pd.read_csv(
            path,
            header=[0, 1],
            index_col=0,
            parse_dates=True
        )
    
    # FileNotFound
    except Exception as e:
        raise DataError(f"Failed to read CSV: {path}") from e

    # Index validation (Must have dates)
    if not isinstance(data.index, pd.DatetimeIndex):
        raise DataError("Index is not DatetimeIndex")

    # Duplicate data
    data = data.sort_index()
    if not data.index.is_unique:
        raise DataError("Duplicate dates in index")

    # Column format mismatch
    if not isinstance(data.columns, pd.MultiIndex):
        raise DataError("Columns are not MultiIndex (Ticker, Field)")

    # Column Mismatch
    if data.columns.nlevels != 2:
        raise DataError("Expected 2-level columns")
    
    # Checking for required fields
    fields = set(data.columns.get_level_values(1).unique())
    if not REQUIRED_FIELDS.issubset(fields):
        raise DataError(f"Missing required fields: {REQUIRED_FIELDS - fields}")

    # Changing values to numbers
    data = data.apply(pd.to_numeric, errors="raise")
    return data