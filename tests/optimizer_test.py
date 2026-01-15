# This script contains a smoke test for all the current OPES optimizers

# Importing modules
import pkgutil
import importlib
import inspect
import pytest
from pathlib import Path

import numpy as np
import pandas as pd

# Finding path of the file and getting parent to extract prices.csv path
BASE_DIR = Path(__file__).resolve().parent
prices_path = BASE_DIR / "prices.csv"

# Statistic fields
STAT_FIELDS = [
    "tickers",
    "weights",
    "portfolio_entropy",
    "herfindahl_index",
    "gini_coefficient",
    "absolute_max_weight",
]


# Function to fetch data from predetermined csv files
# Uses pd.read_csv() method to convert into dataframe
@pytest.fixture(scope="module")
def prices_df():
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    return prices


# Function to discover various optimizer classes within opes.objectives
# We return a list of classes which has the attribute/method 'optimize'
def discover_optimizer_classes():
    import opes.objectives as methods_pkg

    classes = []
    for _, name, _ in pkgutil.iter_modules(methods_pkg.__path__):
        mod = importlib.import_module(f"opes.objectives.{name}")
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                obj.__module__.startswith("opes.objectives")
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


# Function to check the validity of various statistics produced by the optimizer
# Checks entropy, HI, Gini and maxweight validity
def statistics_checker(stats):

    # Error dict
    errors = {}

    # Finite checkinf function
    # Also returns False if x is None
    def is_finite(x):
        return isinstance(x, (int, float, np.floating)) and np.isfinite(x)

    # Entropy: Must be finite and greater than 0
    val = stats.get("portfolio_entropy")
    if not is_finite(val) or val < 0:
        errors["portfolio_entropy"] = f"Invalid entropy: {val}"

    # Herfindahl index: Must be within bounds (0,1]
    val = stats.get("herfindahl_index")
    if not is_finite(val) or not (0 < val <= 1):
        errors["herfindahl_index"] = f"Invalid HI: {val}"

    # Gini coefficient: Must be within bounds [0,1]
    val = stats.get("gini_coefficient")
    if not is_finite(val) or not (0 <= val <= 1):
        errors["gini_coefficient"] = f"Invalid Gini: {val}"

    # Absolute max weight: Must be within bounds (0, 1]
    val = stats.get("absolute_max_weight")
    if not is_finite(val) or not (0 < val <= 1):
        errors["absolute_max_weight"] = f"Invalid max weight: {val}"

    return errors


# Test function on all optimizers discovered
# Optimizes each optimizer with the same data and produces resultant weights
# Nans are also checked
def test_all_optimizers_smoke(prices_df):

    # Checking if optimizers are discovered
    optimizers = discover_optimizer_classes()
    assert optimizers, "No optimizers discovered"

    # creating a failure dictionary to conveniently display which optimizer took the hit
    failures = {}
    stochastic = {}
    stats_fails = {}
    doesnt_sum_to_one = {}

    for cls in optimizers:

        # If the optimizer cannot be instantiated, it is skipped
        # Filters base optimizer ABC
        if not can_instantiate(cls):
            continue

        # ---------- OPTIMIZATION CHECK ----------
        try:
            # Optimizing two times to compare weights
            opt = cls()

            # Checking for weight_bounds, utilizing it if present
            sig = inspect.signature(opt.optimize)
            kwargs = {}

            if "weight_bounds" in sig.parameters:
                kwargs["weight_bounds"] = (-0.5, 1)

            # Optimizing twice
            first_result = opt.optimize(prices_df, **kwargs)
            second_result = opt.optimize(prices_df, **kwargs)

            # Checking if the weights are equal (close) and adding to dictionary of stochastic outputs
            if not np.allclose(first_result, second_result):
                stochastic[cls.__name__] = (first_result, second_result)

            # Checking if the weights sum to 1
            if not np.isclose(first_result.sum(), 1):
                doesnt_sum_to_one[cls.__name__] = f"Does not sum to 1: {first_result}"
        except Exception as e:
            failures[cls.__name__] = str(e)
            continue

        # ---------- STATISTICS CHECK ----------
        try:
            statistics = opt.stats()
            # Adding to stats_fails dictionary if any statistics check fails
            statistics_checker_errors = statistics_checker(statistics)
            if statistics_checker_errors:
                stats_fails[cls.__name__] = (
                    f"Wrong/Missing statistic value(s): {statistics_checker_errors}"
                )
        except Exception as e:
            stats_fails[cls.__name__] = str(e)

        # ---------- RESULT VALIDITY ----------
        assert first_result is not None, f"{cls.__name__} did not produce weights."
        assert not np.any(np.isnan(first_result)), f"{cls.__name__} produced NaNs."
        assert (
            first_result.shape[0] == prices_df.shape[1]
        ), f"{cls.__name__} produced invalid weights shape."
        assert np.all(
            np.isfinite(first_result)
        ), f"{cls.__name__} produced infinite weights."

    # Asserting failures
    assert not failures, "Optimizers failed to compute weights:\n" + "\n".join(
        f"{k}: {v}" for k, v in failures.items()
    )
    assert not stochastic, "Optimizers produced stochastic weights:\n" + "\n".join(
        f"{k}: {v}" for k, v in stochastic.items()
    )
    assert (
        not doesnt_sum_to_one
    ), "Optimizers returned weights that do not sum to 1:\n" + "\n".join(
        f"{k}: {v}" for k, v in doesnt_sum_to_one.items()
    )
    assert not stats_fails, "Optimizers failed to compute statistics:\n" + "\n".join(
        f"{k}: {v}" for k, v in stats_fails.items()
    )
