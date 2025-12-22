# OPES

An open-source Python library for advanced portfolio optimization and backtesting.

## Overview

OPES provides a plethora of quantitative portfolio optimizers with a comprehensive backtesting engine. Test strategies against historical data with configurable slippage costs (stochastic or constant).

## Key Features

- **20+ optimizers**: Mean-Variance, Max Sharpe, Kelly Criterion, Risk Parity, CVaR, Online Learning models and more
- **Advanced backtesting**: Historical performance analysis with wealth plots and comprehensive metrics
- **Stochastic slippage models**: Gamma, Lognormal, Poisson Jump, Inverse Gaussian, or constant costs
- **Flexible regularization**: Entropy, L2, and MaxWeight regularizers
- **Rich metrics**: Sharpe, Sortino, Calmar, Max Drawdown, CVaR, VaR, CAGR, Skewness, Kurtosis and more

## Optimizers

**Utility-Based**: Quadratic Utility, CRRA, CARA, HARA  
**Markowitz**: Mean-Variance, Max Sharpe, GMV, Max Mean
**Heurstic-Based**: Risk Parity, Inverse Volatility Portfolio, 1/N, Max Diversification
**Risk Measures**: CVaR, EVaR, Mean-CVaR, Mean-EVaR
**Online Learning**: Exponential Gradient, BCRP (with regularization)