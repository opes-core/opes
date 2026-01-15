# This script contains a backtester test for all the current OPES optimizers

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


# Function to split train and test data
# Split used within this function is 50:50
def split_train_test(dataframe):
    mid = len(dataframe) // 2
    train = dataframe.iloc[:mid]
    test = dataframe.iloc[mid:]

    return train, test


# Function to validate backtest results
# Validate key availability, length of arrays and other statistical properties
def validate_backtest_results(results):
    issues = {}

    # Check keys
    required_keys = ["returns", "weights", "costs"]
    for key in required_keys:
        if key not in results:
            issues[key] = "Missing key in results dictionary."

    if issues:
        return False, issues

    returns = results["returns"]
    weights = results["weights"]
    costs = results["costs"]

    # Check return existence
    if len(returns) == 0:
        issues["returns_empty"] = "Returns array is empty"

    # Check types
    if not isinstance(returns, np.ndarray):
        issues["returns"] = f"Expected np.ndarray, got {type(returns)}"
    if not isinstance(weights, np.ndarray):
        issues["weights"] = f"Expected np.ndarray, got {type(weights)}"
    if not isinstance(costs, np.ndarray):
        issues["costs"] = f"Expected np.ndarray, got {type(costs)}"

    # Check shapes
    n_steps = len(returns)
    if weights.shape[0] != n_steps:
        issues["weights_shape"] = (
            f"First dimension of weights ({weights.shape[0]}) must match returns length ({n_steps})"
        )
    if costs.shape[0] != n_steps:
        issues["costs_shape"] = (
            f"Costs length ({costs.shape[0]}) must match returns length ({n_steps})"
        )

    # Check values
    if np.isnan(returns).any() or np.isinf(returns).any():
        issues["returns_values"] = "Contains NaN or Inf values"
    if np.isnan(weights).any() or np.isinf(weights).any():
        issues["weights_values"] = "Contains NaN or Inf values"
    if np.isnan(costs).any() or np.isinf(costs).any():
        issues["costs_values"] = "Contains NaN or Inf values"
    if (costs < 0).any():
        issues["costs_values_negative"] = "Costs contain negative values"

    # Check that weights sum to ~1 at each timestep
    weights_sum = np.sum(weights, axis=1)
    if not np.allclose(weights_sum, 1, atol=1e-5):
        issues["weights_sum"] = "Weights do not sum to 1 at all timesteps"

    valid = len(issues) == 0
    return valid, issues


# Function to validate backtest metrics
# Validates key existence and other statistical properties wherever possible
def validate_metrics_output(metrics):
    issues = {}

    # Required keys
    required_keys = {
        "sharpe",
        "sortino",
        "volatility",
        "mean_return",
        "total_return",
        "max_drawdown",
        "var_95",
        "cvar_95",
        "skew",
        "kurtosis",
        "omega_0",
    }

    # Check key presence
    missing = required_keys - metrics.keys()
    extra = metrics.keys() - required_keys

    if missing:
        issues["missing_keys"] = list(missing)
    if extra:
        issues["extra_keys"] = list(extra)

    if issues:
        return False, issues

    # Check value types and finiteness
    for k, v in metrics.items():
        if not isinstance(v, (int, float, np.floating)):
            issues[f"{k}_type"] = f"Expected scalar numeric, got {type(v)}"
            continue

        if np.isnan(v) or np.isinf(v):
            issues[f"{k}_value"] = "Contains NaN or Inf"

    # Logical / domain checks
    if metrics["volatility"] < 0:
        issues["volatility_domain"] = "Volatility cannot be negative"

    if metrics["max_drawdown"] < 0:
        issues["max_drawdown_domain"] = "Max drawdown (loss) should be >= 0"

    if metrics["cvar_95"] < metrics["var_95"]:
        issues["cvar_logic"] = "CVaR should be >= VaR (greater loss)"

    if metrics["omega_0"] < 0:
        issues["omega_domain"] = "Omega ratio cannot be negative"

    valid = len(issues) == 0
    return valid, issues


# Backtester testing function
# Tests both static and rolling backtests
def test_all_optimizers_smoke(prices_df):

    from opes.backtester import Backtester

    # Checking if optimizers are discovered
    optimizers = discover_optimizer_classes()
    assert optimizers, "No optimizers discovered"

    # Obtaining training and testing data
    train, test = split_train_test(prices_df)

    # Creating multiple failure dictionaries for modular mapping
    failures = {}
    failures_static = {}
    failures_rolling = {}
    failures_metrics = {}

    for cls in optimizers:

        # If the optimizer cannot be instantiated, it is skipped
        # Filters base optimizer ABC
        if not can_instantiate(cls):
            continue

        # Initiating backtester
        test_backtester = Backtester(train_data=train, test_data=test)

        # ---------- BACKTEST TESTING ----------
        try:
            # Initiating optimizer and feeding it into backtest
            opt = cls()

            # Executing static backtest
            static_details = test_backtester.backtest(opt)

            # Executing rolling backtest (Daily rebalancing)
            rolling_details = test_backtester.backtest(opt, rebalance_freq=1)

            # Getting static and rolling results
            static_results = validate_backtest_results(static_details)
            rolling_results = validate_backtest_results(rolling_details)

            # Marking static and rolling results if they are flawed
            if not static_results[0]:
                failures_static[cls.__name__] = (
                    f"Failed static backtest: {static_results[1]}"
                )
            if not rolling_results[0]:
                failures_rolling[cls.__name__] = (
                    f"Failed rolling backtest: {rolling_results[1]}"
                )
        except Exception as e:
            failures[cls.__name__] = str(e)
            continue

        # ---------- METRICS TESTING ----------
        returns_array = static_details["returns"]
        try:

            # Obtaining metrics and feeding into validation function
            test_metrics = test_backtester.get_metrics(returns_array)
            test_metrics_results = validate_metrics_output(test_metrics)

            # Marking metrics validation failures
            if not test_metrics_results[0]:
                failures_metrics[cls.__name__] = (
                    f"Failed metrics test: {test_metrics_results[1]}"
                )
        except Exception as e:
            failures[cls.__name__] = str(e)
            continue

    # Asserting failures
    assert not failures, "Method failure:\n" + "\n".join(
        f"{k}: {v}" for k, v in failures.items()
    )
    assert (
        not failures_static
    ), "Optimizers failed static backtest validity check:\n" + "\n".join(
        f"{k}: {v}" for k, v in failures_static.items()
    )
    assert (
        not failures_rolling
    ), "Optimizers failed rolling backtest validity check:\n" + "\n".join(
        f"{k}: {v}" for k, v in failures_rolling.items()
    )
    assert not failures_metrics, "Optimizers metrics validity check:\n" + "\n".join(
        f"{k}: {v}" for k, v in failures_metrics.items()
    )
