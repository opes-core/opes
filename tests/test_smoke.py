# This script contains a smoke test for all the current OPES optimizers
# Since CI is not integrated within the module yet, testing is manual
# So I have used yfinance to fetch data. Integration with CI will result in loading data from csv files

# Importing modules
import pkgutil
import importlib
import inspect
import pytest

import numpy as np
import pandas as pd
import yfinance as yf
import abc

# A stable list of tickers
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN"]


# Function to fetch yfinance data
@pytest.fixture(scope="module")
def returns_df():

    # Fetching yfinance data over a period of 1y
    # This gives a high chance in no nans over the period
    prices = yf.download(
        TICKERS,
        period="1y",
        auto_adjust=True,
        progress=False,
    )["Close"]

    # Computing returns
    # returns are computed using closed prices as mentioned within the documentation
    returns = prices.pct_change().dropna()
    # quick check for shape -> length consistency
    assert returns.shape[1] == len(TICKERS)
    return returns


# Function to discover various optimizer classes within opes.methods
# We return a list of classes which has the attribute/method 'optimize'
def discover_optimizer_classes():
    import opes.methods as methods_pkg

    classes = []
    for _, name, _ in pkgutil.iter_modules(methods_pkg.__path__):
        mod = importlib.import_module(f"opes.methods.{name}")
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                obj.__module__.startswith("opes.methods")
                and hasattr(obj, "optimize")
                and not inspect.isabstract(obj)
            ):
                classes.append(obj)
    return classes


# Function to check if the class can be instantiated
# Since the abstract class Optimizer also possesses the 'optimize' method, this function is a filter
def can_instantiate(cls):
    sig = inspect.signature(cls.__init__)
    for p in list(sig.parameters.values())[1:]:
        if p.default is inspect._empty:
            return False
    return True


# Smoke test function on all optimizers discovered
# Optimizes each optimizer with the same data and produces resultant weights
# Nans are also checked
def test_all_optimizers_smoke(returns_df):

    # Checking if optimizers are discovered
    optimizers = discover_optimizer_classes()
    assert optimizers, "No optimizers discovered"

    # creating a failure dictionary to conveniently display which optimizer took the hit
    failures = {}
    for cls in optimizers:

        # If the optimizer cannot be instantiated, it is skipped
        if not can_instantiate(cls):
            continue

        # Trying to optimize
        # This checks the base, optimization related errors if any
        try:
            opt = cls()
            result = opt.optimize(returns_df)
        except Exception as e:
            failures[cls.__name__] = str(e)
            continue

        # Checking if the result obtained is invalid (is None or produced NaNs)
        assert result is not None
        assert not np.any(np.isnan(result)), f"{cls.__name__} produced NaNs"

    # Raising errors if failures exist, while displaying the optimizer possessing the error
    if failures:
        raise AssertionError(
            "Optimizers failed:\n" + "\n".join(f"{k}: {v}" for k, v in failures.items())
        )
