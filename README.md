# OPES

An Open-source Portfolio Estimation System for advanced portfolio optimization and backtesting.

---

## Overview

OPES is a comprehensive Python library for advanced portfolio optimization and backtesting. It is designed for quantitative finance enthusiasts who wants to explore cutting-edge methods for portfolio research or understanding. OPES aims to be research first, trying to prioritize sheer breadth over depth for each paradigm. OPES aims provide a wide range of portfolio strategies, risk measures and robust evaluation tools for users to always stay updated with current portfolio optimization techniques. OPES's inspiration grew from my own project, MOP, along with the giants of portfolio optimization: PyPortfolioOpt and Riskfolio-Lib.

Visit the [documentation](https://opes.pages.dev) for detailed insights on OPES.

---

## Disclaimer

The information provided by OPES is for educational, research and informational purposes only. It is not intended as financial, investment or legal advice. Users should conduct their own due diligence and consult with licensed financial professionals before making any investment decisions. OPES and its contributors are not liable for any financial losses or decisions made based on this content. Past performance is not indicative of future results.

---

## Portfolio Objectives

| Classification                           | Name                                               |
| ---------------------------------------- | -------------------------------------------------- |
| **Utility Theory**                       | Quadratic Utility                                  |
|                                          | Constant Relative Risk Aversion (CRRA)             |
|                                          | Constant Absolute Risk Aversion (CARA)             |
|                                          | Hyperbolic Absolute Risk Aversion (HARA)           |
|                                          | Kelly Criterion & Fractional Kelly                 |
| **Markowitz Paradigm**                   | Maximum Mean Return                                |
|                                          | Minimum Variance                                   |
|                                          | Mean-Variance                                      |
|                                          | Maximum Sharpe Ratio                               |
| **Principled Heuristics**                | Uniform (1/N)                                      |
|                                          | Risk Parity                                        |
|                                          | Inverse Volatility                                 |
|                                          | Softmax Mean                                       |
|                                          | Maximum Diversification                            |
|                                          | Return Entropy Portfolio Optimization              |
| **Risk Measures**                        | Value at Risk (VaR)                                |
|                                          | Conditional Value at Risk (CVaR)                   |
|                                          | Mean-CVaR                                          |
|                                          | Entropic Value at Risk (EVaR)                      |
|                                          | Mean-EVaR                                          |
|                                          | Entropic Risk Measure                              |
|                                          | Worst-Case Loss                                    |
| **Online Learning**                      | Cover's Universal Portfolios                       |
|                                          | Best Constant Rebalanced Portfolio (BCRP)          |
|                                          | Exponential Gradient                               |
| **Distributionally Robust Optimization** | KL-Ambiguity Robust Maximum Mean Return            |
|                                          | KL-Ambiguity Robust Kelly                          |
|                                          | KL-Ambiguity Robust Fractional Kelly               |
|                                          | Wasserstein-Ambiguity Robust Maximum Mean Return   |
|                                          | Wasserstein-Ambiguity Robust Minimum Variance      |
|                                          | Wasserstein-Ambiguity Robust Mean-Variance         |

## Metrics

| Portfolio Metrics        | Backtest Metrics             |
| ------------------------ | ---------------------------- |
| Tickers                  | Sharpe                       |
| Weights                  | Sortino                      |
| Portfolio Entropy        | Volatility                   |
| Herfindahl Index         | Average Return               |
| Gini Coefficient         | Total Return                 |
| Absolute Maximum Weight  | Mean Drawdown                |
|                          | Maximum Drawdown             |
|                          | Geometric Growth Rate        |
|                          | Value-at-Risk 95             |
|                          | Conditional-Value-at-Risk 95 |
|                          | Skew                         |
|                          | Kurtosis                     |
|                          | Omega Ratio                  |
|                          | Ulcer Index                  |
|                          | Hit Ratio                    |

## Other Features

| Slippage Models            | Regularization Schemes                    |
| -------------------------- | ----------------------------------------- |
| Constant                   | L1 Regularization                         |
| Gamma                      | L2 Regularization                         |
| Lognormal                  | L-infinity Regularization                 |
| Inverse Gaussian           | Entropy                                   |
| Compound Poisson-Lognormal | Weight Variance                           |
|                            | Mean Pairwise Absolute Deviation          |
|                            | KL-Divergence from Uniform (Experimental) |
|                            | JS-Divergence from Uniform (Experimental) |

---

## Installation

### Using `pip`

If you have `pip`, it is very convenient to install `opes`.

```bash
pip install opes
```

### From the Source

Alternatively, you are also welcome to install directly from the GitHub repository.

```bash
git clone https://github.com/opes-core/opes.git
cd opes
pip install .
```

You can also install in editable mode if you plan on making any changes to the source code.

```bash 
# After cloning
pip install -e .
```

### Verification

Verify your installation by using `pip`.

```bash
pip show opes
```

You can also verify by using python.

```python
>>> import opes
>>> opes.__version__
```

---

## Getting Started

`opes` is designed to be minimalistic and easy to use and learn for any user. Here is an example script which implements my favorite portfolio, the Kelly Criterion.

```python
# I recommend you use yfinance for testing.
# However for serious research, using an external, faster API would be more fruitful.
import yfinance as yf

# Importing our Kelly class
from opes.objectives.utility_theory import Kelly

# Obtaining ticker data
# Basic yfinance stuff
TICKERS = ["AAPL", "NVDA", "PFE", "TSLA", "BRK-B", "SHV", "TLT"]
asset_data = yf.download(
    tickers=TICKERS, 
    start="2010-01-01", 
    end="2020-01-01", 
    group_by="ticker", 
    auto_adjust=True
)

# Initialize a Kelly portfolio with fractional exposure and L2 regularization
# Fractional exposure produces less risky weights and L2 regularization contributes in penalizing concentration
kelly_portfolio = Kelly(fraction=0.8, reg="l2", strength=0.001)

# Compute portfolio weights with custom weight bounds
kelly_portfolio.optimize(data, weight_bounds=(0.05, 0.8))

# Clean negligible allocations
cleaned_weights = kelly_portfolio.clean_weights(threshold=1e-6)

# Output the final portfolio weights
print(cleaned_weights)
```

This showcases the simplicty of the module. However there are far more diverse features you can still explore. If you're looking for more examples, preferably some of them with *"context"*, I recommend you check out the [examples](https://opes.pages.dev/examples/good_strategy/) page within the documentation.

---

## Testing

Tests for `opes` are written using the `pytest` module. You can run the tests easily using the following command.

```bash
cd project-root    # Navigate to project root
pytest
```

This will run three scripts, each dedicated to testing the optimizer, regularizer and backtesting engine. Note that the tests are heavy and can take a significant amount of time since it tests each of the available objectives. We test using a year's data for the following tickers.

```
GOOG, AAPL, AMZN, MSFT
```

The price data is stored in the `prices.csv` file within the `tests/` directory. The number of tickers are limited to 4 since there are computationally heavy portfolio objectives (like `UniversalPortfolios`) included which may take an eternity to test well using multiple tickers.

Also it eats up RAM like pac-man.

---

## Upcoming Features (Unconfirmed)

These features are still in the works and may or may not appear in later updates:

| **Objective Name (Category)**                    |
| ------------------------------------------------ |
| Hierarchical Risk Parity (Principled Heuristics) |              
| Online Newton Step (Online Learning)             | 
| ADA-BARRONS (Online Learning)                    |
