# OPES

An Open-source Portfolio Estimation System for advanced portfolio optimization and backtesting.

---

## Overview

OPES provides a plethora of quantitative portfolio optimizers with a comprehensive backtesting engine. Test strategies against historical data with configurable slippage costs.

For a quick documentation on this module, visit [opes-documentation](https://github.com/opes-core/opes-documentation)

For a detailed guide on this module, visit [opes-book.vercel.app](https://opes-book.vercel.app)

---

## Disclaimer

The information provided by OPES is for educational, research and informational purposes only. It is not intended as financial, investment or legal advice. Users should conduct their own due diligence and consult with licensed financial professionals before making any investment decisions. OPES and its contributors are not liable for any financial losses or decisions made based on this content. Past performance is not indicative of future results.

---

## Porfolio Objectives

### Utility Theory
- Quadratic Utility
- Constant Relative Risk Aversion
- Constant Absolute Risk Aversion
- Hyperbolic Absolute Risk Aversion
- Kelly Criterion and Fractions

### Markowitz Paradigm
- Maximum Mean
- Minimum Variance
- Mean Variance
- Maximum Sharpe

### Principled Heuristics
- 1/N
- Risk Parity
- Inverse Volatility
- Softmax Mean
- Maximum Diversification
- Return Entropy Portfolio Optimization

### Risk Measures
- Conditional Value at Risk
- Mean-CVaR
- Entropic Value at Risk
- Mean-EVaR
- Entropic Risk Measure

### Online Learning
- BCRP with weight regularization (FTL/FTRL support)
- Exponential Gradient

### Distributionally Robust Optimization
- KL-Ambiguity Distributionally Robust Maximum Mean
- KL-Ambiguity Distributionally Robust Kelly and Fractions
- Wasserstein-Ambiguity Distributionally Robust Maximum Mean

## Slippage Models
- Constant
- Gamma
- Lognormal
- Inverse Gaussian
- Compound Poisson-Lognormal

## Regularization Schemes
- L1
- L2
- L-infinity
- Entropy
- Weight Variance
- Mean Pairwise Absolute Deviation

## Backtest Metrics
 - Sharpe Ratio
 - Sortino Ratio
 - Volatility
 - Average Return
 - Total Return
 - Maximum Drawdown
 - Value at Risk 95
 - Conditional Value at Risk 95
 - Skew
 - Kurtosis
 - Omega Ratio

## Portfolio Metrics
- Tickers
- Weights
- Portfolio Entropy
- Herfindahl Index
- Gini Coefficient
- Absolute Maximum Weight

---

## Upcoming Features (Unconfirmed)

These features are still in the works and may or may not appear in later updates:

* Mean–Variance–Skew–Kurtosis Optimization (Markowitz)
* Hierarchical Risk Parity (Principled Heuristics)
* Online Newton Step (Online Learning)
* Ada Barrons (Online Learning)
* Wasserstein Ambiguity Duals (Distributionally Robust)

  * Global Minimum Variance (GMV)
  * Mean–Variance Optimization (MVO)
  * Kelly Criterion
