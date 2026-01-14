# OPES

An Open-source Portfolio Estimation System for advanced portfolio optimization and backtesting.

---

## Overview

OPES is a comprehensive Python library for advanced portfolio optimization and backtesting. Designed for quantitative finance enthusiasts, OPES provides a wide range of portfolio strategies, risk measures and robust evaluation tools.

Visit the [documentation](https://opes.pages.dev) for a detailed walkthrough on this module.

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
|                                          | Mean–Variance                                      |
|                                          | Maximum Sharpe Ratio                               |
| **Principled Heuristics**                | Uniform (1/N)                                      |
|                                          | Risk Parity                                        |
|                                          | Inverse Volatility                                 |
|                                          | Softmax Mean                                       |
|                                          | Maximum Diversification                            |
|                                          | Return Entropy Portfolio Optimization              |
| **Risk Measures**                        | Conditional Value at Risk (CVaR)                   |
|                                          | Mean–CVaR                                          |
|                                          | Entropic Value at Risk (EVaR)                      |
|                                          | Mean–EVaR                                          |
|                                          | Entropic Risk Measure                              |
| **Online Learning**                      | Best Constant Rebalanced Portfolio (BCRP)          |
|                                          | Exponential Gradient                               |
| **Distributionally Robust Optimization** | KL-Ambiguity Robust Maximum Mean Return            |
|                                          | KL-Ambiguity Robust Kelly                          |
|                                          | KL-Ambiguity Robust Fractional Kelly               |
|                                          | Wasserstein-Ambiguity Robust Maximum Mean Return   |

## Metrics

| Portfolio Metrics        | Backtest Metrics             |
| ------------------------ | ---------------------------- |
| Tickers                  | Sharpe                       |
| Weights                  | Sortino                      |
| Portfolio Entropy        | Volatility                   |
| Herfindahl Index         | Average Return               |
| Gini Coefficient         | Total Return                 |
| Absolute Maximum Weight  | Maximum Drawdown             |
|                          | Value-at-Risk 95             |
|                          | Conditional-Value-at-Risk 95 |
|                          | Skew                         |
|                          | Kurtosis                     |
|                          | Omega Ratio                  |

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

## Upcoming Features (Unconfirmed)

These features are still in the works and may or may not appear in later updates:

* Worst-Case Loss Optimization (Risk Measures)
* Value-at-Risk (Risk Measures)
* Hierarchical Risk Parity (Principled Heuristics)
* Universal Portfolios (Online Learning)
* Online Newton Step (Online Learning)
* ADA-BARRONS (Online Learning)
* Wasserstein Ambiguity Duals (Distributionally Robust)

  * Global Minimum Variance (GMV)
  * Mean–Variance Optimization (MVO)