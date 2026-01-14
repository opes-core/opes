# This script contains a smoke test for all available regularizers

# Importing modules
import pytest
from pathlib import Path

import numpy as np
import pandas as pd

# Finding path of the file and getting parent to extract prices.csv path
BASE_DIR = Path(__file__).resolve().parent
prices_path = BASE_DIR / "prices.csv"

# Available regularizers
# L1 regularizer is checked separately since it is primarily intended for long-short portfolios
regularizers_available = ["l2", "l-inf", "entropy", "variance", "mpad", "kld", "jsd"]


# Function to fetch data from predetermined csv files
# Uses pd.read_csv() method to convert into dataframe
@pytest.fixture(scope="module")
def prices_df():
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    return prices


# Weight checking function
def weight_check(w_base, w_reg):
    # We check for change in weights of w_reg from w_base
    # However, if w_base is equal weighted (RARE CASE), then regularizer penalty is already minimized at w_base.

    # Generating equal weights
    equal_w = np.ones(len(w_base)) / len(w_base)

    # If equal weights, then the regularizer may have no affect, since equal weights minimizes the penalty
    if not np.allclose(w_base, equal_w):

        # Returning if the regularizer has any affect
        return not np.allclose(w_base, w_reg)
    else:
        # If the base weights are equal, then the regularization is minimized
        # So default True is returned
        return True


def test_all_regularizers(prices_df):

    # Importing one SLSQP based optimizer and one differential_evolution based optimizer
    from opes.objectives.markowitz import MaxMean, MaxSharpe

    # Initiating failures dictionary
    failures = {}

    # Computing base optimizer weights
    base_maxmean_weights = MaxMean().optimize(prices_df)
    base_maxsharpe_weights = MaxSharpe().optimize(prices_df)

    # Optimizing for each regularizer
    for regularizer in regularizers_available:

        # Maximum Mean optimization
        opt = MaxMean(reg=regularizer, strength=10)
        maxmean_weights = opt.optimize(data=prices_df)

        # Maximum Sharpe optimization
        opt = MaxSharpe(reg=regularizer, strength=10)
        maxsharpe_weights = opt.optimize(data=prices_df)

        # Comparing weights and adding to dict if necessary
        if not weight_check(base_maxmean_weights, maxmean_weights):
            failures[regularizer] = "FAILED WEIGHT CHECK ON MAXMEAN."
            continue
        if not weight_check(base_maxsharpe_weights, maxsharpe_weights):
            failures[regularizer] = "FAILED WEIGHT CHECK ON MAXSHARPE."

    # Separately checking for L1
    base_maxmean_longshort = MaxMean().optimize(prices_df, weight_bounds=(-0.5, 1))
    reg_maxmean_longshort = MaxMean(reg="l1", strength=10).optimize(
        prices_df, weight_bounds=(-0.5, 1)
    )
    if not weight_check(base_maxmean_longshort, reg_maxmean_longshort):
        failures["l1"] = "FAILED WEIGHT CHECK ON MAXMEAN."

    # Displaying failures if any
    assert not failures, "Regularizers failed weight check:\n" + "\n".join(
        f"{k}: {v}" for k, v in failures.items()
    )
