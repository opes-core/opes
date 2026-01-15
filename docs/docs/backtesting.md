# `backtester`

OPES contains a portfolio backtesting engine inclusive of a simple plotting function
using a supplied optimizer. The method supports both static and rolling portfolio
construction, applies transaction costs and enforces strict no-lookahead
constraints during rebalancing.

The backtest operates exclusively on the test dataset while ensuring that
all portfolio decisions are made using only information available at the time of execution.

---





## `Backtester`

```python
class Backtester(train_data=None, test_data=None, cost={'const': 10.0})
```

A comprehensive backtesting engine for financial time series.

This class manages training and testing datasets, ensuring that
any missing values (NaNs) are removed for robust backtesting loops.
It also stores transaction cost parameters for portfolio simulations.

**Args:**

- `train_data` (*pd.DataFrame*): Historical training data. Defaults to None.
- `test_data` (*pd.DataFrame*): Historical testing data. Defaults to None.
- `cost` (*dict, optional*): Transaction cost parameters. Defaults to `{'const': 10.0}`. Various cost models are given below:
    - `{'const': constant_bps_value}`
    - `{'gamma': (shape, scale)}`
    - `{'lognormal': (mu, sigma)}`
    - `{'inversegaussian': (mean, shape)}`
    - `{'jump': (arrival_rate, mu, sigma)}`

!!! note "Notes:"
    - `train_data` and `test_data` must be of the same format: either Single-Index with `DateTimeIndex` and tickers as columns or Multi-Index with a necessary "Close" column in level 1 and tickers in level 0. Examples are shown below
    ```
    # Single-Index Example
    Ticker           TSLA      NVDA       GME        PFE       AAPL  ...
    Date
    2015-01-02  14.620667  0.483011  6.288958  18.688917  24.237551  ...
    2015-01-05  14.006000  0.474853  6.460137  18.587513  23.554741  ...
    2015-01-06  14.085333  0.460456  6.268492  18.742599  23.556952  ...
    2015-01-07  14.063333  0.459257  6.195926  18.999102  23.887287  ...
    2015-01-08  14.041333  0.476533  6.268492  19.386841  24.805082  ...
    ...

    # Multi-Index Example Structure (OHLCV)
    Columns:
    + Ticker (e.g. GME, PFE, AAPL, ...)
      + Open
      + High
      + Low
      + Close
      + Volume
    ```
    - NaN values in both `train_data` and `test_data` are automatically dropped to prevent indexing errors during backtests. So Assets with shorter trading histories truncate the dataset.
    - After cleaning/truncation, close prices are extracted per asset. Returns are computed as:

    $$R_t = \frac{P^{(t)}}{P^{(t-1)}} - 1$$

---

### Methods

#### `backtest`

```python
def backtest(
    optimizer,
    rebalance_freq=None,
    seed=None,
    weight_bounds=None,
    clean_weights=False
)
```

Execute a portfolio backtest over the test dataset using a given optimizer.

This method performs either a static-weight backtest or a rolling-weight
backtest depending on whether `rebalance_freq` is specified. It also
applies transaction costs and ensures no lookahead bias during rebalancing.
For a rolling backtest, any common date values are dropped, the first occurrence
is considered to be original and kept.

!!! warning "Warning:"
    Some online learning methods such as `ExponentialGradient` update weights based
    on the most recent observations. Setting `rebalance_freq` to any value other
    than `1` (or possibly `None`) may result in suboptimal performance, as
    intermediate data points will be ignored and not used for weight updates.
    Proceed with caution when using other rebalancing frequencies with online learning algorithms.

**Args**

- `optimizer`: Portfolio optimizer object with an `optimize` method.
- `rebalance_freq` (*int or None, optional*): Frequency of rebalancing in time steps. If `None`, a static weight backtest is performed. Defaults to `None`.
- `seed` (*int or None, optional*): Random seed for reproducible cost simulations. Defaults to `None`.
- `weight_bounds` (*tuple, optional*): Bounds for portfolio weights passed to the optimizer if supported.


**Returns:**

- `dict`: Backtest results containing the following keys:
    - `'returns'` (*np.ndarray*): Portfolio returns after accounting for costs.
    - `'weights'` (*np.ndarray*): Portfolio weights at each timestep.
    - `'costs'` (*np.ndarray*): Transaction costs applied at each timestep.

**Raises**

- `DataError`: If the optimizer does not accept weight bounds but `weight_bounds` are provided.
- `PortfolioError`: If input validation fails (via `_backtest_integrity_check`).


!!! note "Notes:"
    - All returned arrays are aligned in time and have length equal to the test dataset.
    - Static weight backtest: Uses a single set of optimized weights for all test data. This denotes a constant rebalanced portfolio.
    - Rolling weight backtest: Re-optimizes weights at intervals defined by `rebalance_freq` using only historical data up to the current point to prevent lookahead bias.
    - Transaction costs are applied using the `slippage()` function, supporting various cost models.
    - Returns and weights are stored in arrays aligned with test data indices.

!!! example "Example:"
    ```python
    import numpy as np

    # Importing necessary OPES modules
    from opes.objectives.utility_theory import Kelly
    from opes.backtester import Backtester

    # Place holder for your price data
    from some_random_module import trainData, testData

    # Declare train and test data
    training = trainData()
    testing = testData()

    # Declaring kelly
    kelly_optimizer = Kelly(fraction=0.8)

    # Initializing Backtest with constant costs
    tester = Backtester(train_data=training, test_data=testing)

    # Obtaining backtest data for kelly optimizer
    kelly_backtest = tester.backtest(optimizer=kelly_optimizer, rebalance_freq=21)

    # Printing results
    for key in kelly_backtest:
      print(f"{key}: {kelly_backtest}")
    ```

---

#### `get_metrics`

```python
def get_metrics(returns)
```

Computes a comprehensive set of portfolio performance metrics from returns.

This method calculates risk-adjusted and absolute performance measures
commonly used in finance, including volatility, drawdowns, and tail risk metrics.

**Args**

- `returns` (*array-like*): Array or list of periodic portfolio returns. Will be converted to numpy array.


**Returns:**

- `dict`: Dictionary of performance metrics with the following keys:
    - `'sharpe'`: Sharpe ratio.
    - `'sortino'`: Sortino ratio.
    - `'volatility'`: Standard deviation of returns (%).
    - `'mean_return'`: Mean return (%).
    - `'total_return'`: Total cumulative return (%).
    - `'max_drawdown'`: Maximum drawdown.
    - `'var_95'`: Value at Risk at 95% confidence level.
    - `'cvar_95'`: Conditional Value at Risk (expected shortfall) at 95%.
    - `'skew'`: Skewness of returns.
    - `'kurtosis'`: Kurtosis of returns.
    - `'omega_0'`: Omega ratio (gain/loss ratio).

!!! note "Notes:"
    - Volatility, mean return, total return, maximum drawdown, VaR and CVaR are scaled to percentages.
    - VaR, CVaR and maximum drawdown are returned as loss values (usually positive).
    - Tail risk metrics (VaR, CVaR) are based on the lower 5% of returns.
    - Returns should be cleaned (NaNs removed) before passing to this method.

!!! example "Example:"
    ```python
    # Importing portfolio method and backtester
    from opes.objectives.markowitz import MaxSharpe
    from opes.backtester import Backtester

    # Place holder for your price data
    from some_random_module import trainData, testData

    # Declare train and test data
    training = trainData()
    testing = testData()

    # Declare the maximum sharpe optimizer with risk-free rate of 0.02
    max_sharpe_opt = MaxSharpe(risk_free=0.02)

    # Initializing Backtest with constant costs
    tester = Backtester(train_data=training, test_data=testing)

    # Obtaining returns array from backtest and finding metrics
    optimizer_returns = tester.backtest(optimizer=max_sharpe_opt)['returns']
    metrics = tester.get_metrics(optimizer_returns)

    # Printing sharpe and maximum drawdown
    print(metrics["sharpe"], metrics["max_drawdown"])
    ```

---

#### `plot_wealth`

```python
def plot_wealth(returns_dict, initial_wealth=1.0, savefig=False)
```

OPES ships with a basic plotting utility for visualizing portfolio wealth over time.

This method exists for quick inspection and debugging, not for deep performance analysis.
It visualizes cumulative wealth for one or multiple strategies
using their periodic returns. It also provides a breakeven reference line
and optional saving of the plot to a file.

!!! tip "Recommendation:"
    For serious research, reporting, or strategy comparison, we strongly recommend writing your own custom plotting pipeline.
    Real evaluation usually needs rolling Sharpe, drawdowns, volatility regimes, benchmark overlays and other diagnostics that
    go far beyond a single equity curve.

**Args**

- `returns_dict` (*dict or np.ndarray*): Dictionary of strategy names to returns arrays or a single numpy array (treated as one strategy).
- `initial_wealth` (*float, optional*): Starting wealth for cumulative calculation. Defaults to `1.0`.
- `savefig` (*bool, optional*): If `True`, saves the plot as a PNG file with a timestamped filename. Defaults to `False`.


!!! note "Notes:"
    - Converts a single numpy array input into a dictionary with key "Strategy".
    - Computes cumulative wealth as $W_t = W_0 \prod_{i}^T(1+r_i)$.
    - Plots each strategy's wealth trajectory on a logarithmic y-axis.
    - Adds a horizontal breakeven line at the initial wealth.
    - Displays the plot and optionally saves it to a PNG file.

!!! example "Example:"
    ```python
    # Importing portfolio methods and backtester
    from opes.objectives.markowitz import MaxMean, MeanVariance
    from opes.backtester import Backtester

    # Place holder for your price data
    from some_random_module import trainData, testData

    # Declare train and test data
    training = trainData()
    testing = testData()

    # Declare two optimizers
    maxmeanl2 = MaxMean(reg="l2", strength=0.001)
    mvo1_5 = MeanVariance(risk_aversion=1.5)

    # Initializing Backtest with constant costs
    tester = Backtester(train_data=training, test_data=testing)

    # Obtaining returns array from backtest for both optimizers (Monthly Rebalancing)
    scenario_1 = tester.backtest(optimizer=maxmeanl2, rebalance_freq=21)['returns']
    scenario_2 = tester.backtest(optimizer=mvo1_5, rebalance_freq=21)['returns']

    # Plotting wealth
    tester.plot_wealth(
        {
            "Maximum Mean (L2, 1e-3)": scenario_1,
            "Mean Variance (RA=1.5)": scenario_2,
        }
    )
    ```


---

