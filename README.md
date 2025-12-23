# OPES

An open-source Python library for advanced portfolio optimization and backtesting.

## Overview

OPES provides a plethora of quantitative portfolio optimizers with a comprehensive backtesting engine. Test strategies against historical data with configurable slippage costs (stochastic or constant).

## Key Features

- **15+ optimizers (and more to come)**: Mean-Variance, Max Sharpe, Kelly Criterion, Risk Parity, CVaR, Online Learning models and more
- **Advanced backtesting**: Historical performance analysis with wealth plots and comprehensive metrics
- **Stochastic slippage models**: Gamma, Lognormal, Poisson Jump, Inverse Gaussian, or constant costs
- **Flexible regularization**: Entropy, L2, and MaxWeight regularizers
- **Rich metrics**: Sharpe, Sortino, Calmar, Max Drawdown, CVaR, VaR, CAGR, Skewness, Kurtosis and more

## Portfolio Methods

### Utility Theory
- Quadratic Utility
- Constant Relative Risk Aversion
- Constant Absolute Risk Aversion
- Hyperbolic Absolute Risk Aversion
- Kelly Criterion and fractions

### Markowitz Paradigm
- Maximum Mean
- Minimum Variance
- Mean Variance
- Maximum Sharpe

### Principled Heuristics
- Risk Parity
- Inverse Volatility
- Softmax Mean
- Maximum Diversification
- 1/N

### Risk Measures
- CVaR
- Mean-CVaR
- EVaR
- Mean-EvaR

### Online Learning
- BCRP with regularization (FTL/FTRL)
- Exponential Gradient