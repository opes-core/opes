# **Home**

---

## Introduction

OPES is a Python library for advanced portfolio construction, estimation and backtesting. Its core principle is to provide a broad collection of portfolio allocation schemes developed over decades of financial research, combined with flexible regularization mechanisms, standardized portfolio metrics and a robust backtesting engine.

Most modern portfolio estimation libraries are designed primarily for practitioners and professional quantitative researchers. While some user-friendly tools exist, portfolio theory remains fragmented and backtesting methodologies vary significantly across libraries, making fair comparison difficult.

OPES addresses this gap by standardizing portfolio construction workflows and evaluation procedures. It enables consistent comparison of strategies ranging from classical Markowitz-style optimization to modern, distributionally robust and online learning–based portfolio schemes.

---

## Purpose

OPES is a research-oriented and experimentation-focused Python module for portfolio estimation. It supports both deterministic and stochastic analysis, enabling systematic evaluation of a wide range of portfolio allocation approaches. OPES is not a trading bot, a signal generation system or a financial advisor and it does not provide investment recommendations. Its sole purpose is to facilitate rigorous analysis and comparison of portfolio strategies within a controlled and reproducible environment.

---

## Example Snippet

!!! example "Demo"
    ```python
    # Demonstration of portfolio optimization using the Kelly Criterion
    # 'data' represents OHLCV market data grouped by ticker symbols
    
    from opes.methods.utility_theory import Kelly
    
    # Initialize a Kelly portfolio with fractional exposure and L2 regularization
    kelly_portfolio = Kelly(fraction=0.8, reg="l2", strength=0.01)
    
    # Compute portfolio weights with custom bounds and clean negligible allocations
    kelly_portfolio.optimize(data, weight_bounds=(0.05, 0.8))
    cleaned_weights = kelly_portfolio.clean_weights(threshold=1e-6)
    
    # Output the final portfolio weights
    print(cleaned_weights)
    ```