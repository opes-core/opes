# `heuristics`

While utility theory and Markowitz optimization boast mathematical elegance,
the real-world of quantitative investing often prefers simpler heuristics because
estimating returns and covariances accurately from noisy data is basically guessing with style.
Heuristic portfolios dodge this problem by embracing simplicity, with some relying on trader
intuition like equal-weighting assets when *"who knows what will win"* and others using principled
rules such as spreading risk evenly or minimizing portfolio entropy. The common thread is
robustness, as avoiding strong assumptions about return predictions often leads these approaches
to outperform their theoretically optimal cousins outside the textbook.

---





## `InverseVolatility`

```python
class InverseVolatility()
```

Inverse Volatility portfolio.

The inverse volatility portfolio, a practical simplification of minimum
variance strategies used by practitioners since at least the 1970s,
weights assets inversely proportional to their volatilities. The approach
is grounded in the intuition that higher-risk assets should receive
smaller allocations, which is equivalent to risk parity when all assets
are uncorrelated. While deliberately naive about correlations, inverse
volatility portfolios are trivial to compute, require minimal data and
often perform surprisingly well out-of-sample.

The `InverseVolatility` optimizer does not require any parameters to initialize.

### Methods

#### `clean_weights`

```python
def clean_weights(threshold=1e-08)
```

Cleans the portfolio weights by setting very small positions to zero.

Any weight whose absolute value is below the specified `threshold` is replaced with zero.
This helps remove negligible allocations while keeping the array structure intact. This method 
requires portfolio optimization (`optimize()` method) to take place for `self.weights` to be
defined other than `None`.

!!! warning "Warning:"
    This method modifies the existing portfolio weights in place. After cleaning, re-optimization
    is required to recover the original weights.

**Args**

- `threshold` (*float, optional*): Float specifying the minimum absolute weight to retain. Defaults to `1e-8`.


**Returns:**

- `numpy.ndarray`: Cleaned and re-normalized portfolio weight vector.

**Raises**

- `PortfolioError`: If weights have not been calculated via optimization.


!!! note "Notes:"
    - Weights are cleaned using absolute values, making this method compatible with long-short portfolios.
    - Re-normalization ensures the portfolio remains properly scaled after cleaning.
    - Increasing threshold promotes sparsity but may materially alter the portfolio composition.

#### `optimize`

```python
def optimize(data=None, **kwargs)
```

Satisfies the Inverse Volatility objective:

$$
\mathbf{w}_i = \frac{1/\sigma_i}{\sum_j^N 1/\sigma_j}
$$

!!! note "Note"
    Asset weight bounds are defaulted to (0,1).

**Args**

- `data` (*pd.DataFrame*): Ticker price data in either multi-index or single-index formats. Examples are given below:
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
  - Open
  - High
  - Low
  - Close
  - Volume
  ```
- `**kwargs` (*optional*): Included for interface consistency, allowing the backtesting engine to pass additional or optimizer-specific arguments that may be safely ignored by this optimizer.


**Returns:**

- `np.ndarray`: Vector of optimized portfolio weights.

**Raises**

- `DataError`: For any data mismatch during integrity check.


!!! example "Example:"
    ```python
    # Importing the Inverse Volatility Portfolio (IVP) module
    from opes.objectives.heuristics import InverseVolatility as IVP

    # Let this be your ticker data
    training_data = some_data()

    # Initialize portfolio
    inverse_vol = IVP()

    # Optimize portfolio
    weights = inverse_vol.optimize(data=training_data)
    ```

#### `stats`

```python
def stats()
```

Calculates and returns portfolio concentration and diversification statistics.

These statistics help users to inspect portfolio's overall concentration in
allocation. For the method to work, the optimizer must have been initialized, i.e.
the `optimize()` method should have been called at least once for `self.weights`
to be defined other than `None`.

**Returns:**

- A `dict` containing the following keys:
    - `'tickers'` (*list*): A list of tickers used for optimization.
    - `'weights'` (*np.ndarry*): Portfolio weights, output from optimization.
    - `'portfolio_entropy'` (*float*): Shannon entropy computed on portfolio weights.
    - `'herfindahl_index'` (*float*): Herfindahl Index value, computed on portfolio weights.
    - `'gini_coefficient'` (*float*): Gini Coefficient value, computed on portfolio weights.
    - `'absolute_max_weight'` (*float*): Absolute maximum allocation for an asset.

**Raises**

- `PortfolioError`: If weights have not been calculated via optimization.


!!! note "Notes:"
    - All statistics are computed on absolute normalized weights (within the simplex), ensuring compatibility with long-short portfolios.
    - This method is diagnostic only and does not modify portfolio weights.
    - For meaningful interpretation, use these metrics in conjunction with risk and performance measures.

## `MaxDiversification`

```python
class MaxDiversification(reg=None, strength=1)
```

Maximum Diversification optimization.

The maximum diversification portfolio, introduced by Yves Choueifaty and Yves Coignard,
maximizes the diversification ratio, defined as the weighted average of asset volatilities
divided by portfolio volatility. Motivated by the low-volatility anomaly, the strategy
explicitly targets portfolios where diversification benefits are strongest, meaning
portfolio volatility is significantly lower than the average individual volatility. It
relies only on the covariance matrix and naturally yields well-diversified portfolios
without extreme concentration, offering a practical alternative to minimum variance while
preserving broad asset exposure.

**Args**

- `reg` (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
- `strength` (*float, optional*): Strength of the regularization. Defaults to `1`.


### Methods

#### `clean_weights`

```python
def clean_weights(threshold=1e-08)
```

Cleans the portfolio weights by setting very small positions to zero.

Any weight whose absolute value is below the specified `threshold` is replaced with zero.
This helps remove negligible allocations while keeping the array structure intact. This method 
requires portfolio optimization (`optimize()` method) to take place for `self.weights` to be
defined other than `None`.

!!! warning "Warning:"
    This method modifies the existing portfolio weights in place. After cleaning, re-optimization
    is required to recover the original weights.

**Args**

- `threshold` (*float, optional*): Float specifying the minimum absolute weight to retain. Defaults to `1e-8`.


**Returns:**

- `numpy.ndarray`: Cleaned and re-normalized portfolio weight vector.

**Raises**

- `PortfolioError`: If weights have not been calculated via optimization.


!!! note "Notes:"
    - Weights are cleaned using absolute values, making this method compatible with long-short portfolios.
    - Re-normalization ensures the portfolio remains properly scaled after cleaning.
    - Increasing threshold promotes sparsity but may materially alter the portfolio composition.

#### `optimize`

```python
def optimize(
    data=None,
    custom_cov=None,
    seed=100,
    **kwargs
)
```

Solves the Maximum Diversification objective:

$$
\min_{\mathbf{w}} \ -\frac{\mathbf{w}^\top \sigma}{\sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}} + \lambda R(\mathbf{w})
$$

!!! warning "Warning"
    Since the maximum diversification objective is generally non-convex, SciPy's `differential_evolution` optimizer
    is used to obtain more robust solutions. This approach incurs significantly higher computational cost and should
    be used with care.

!!! note "Note"
    Asset weight bounds are defaulted to (0,1).

**Args**

- `data` (*pd.DataFrame*): Ticker price data in either multi-index or single-index formats. Examples are given below:
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
  - Open
  - High
  - Low
  - Close
  - Volume
  ```
- `custom_cov` (*None or array-like of shape (n_assets, n_assets), optional*): Custom covariance matrix. Can be used to inject externally generated covariance matrices (eg. Ledoit-Wolf). Defaults to `None`.
- `seed` (*int or None, optional*): Seed for differential evolution solver. Defaults to `100` to preserve deterministic outputs.
- `**kwargs` (*optional*): Included for interface consistency, allowing the backtesting engine to pass additional or optimizer-specific arguments that may be safely ignored by this optimizer.


**Returns:**

- `np.ndarray`: Vector of optimized portfolio weights.

**Raises**

- `DataError`: For any data mismatch during integrity check.
- `OptimizationError`: If `differential_evolution` solver fails to solve.


!!! example "Example:"
    ```python
    # Importing the maximum diversification module
    from opes.objectives.heuristics import MaxDiversification

    # Let this be your ticker data
    training_data = some_data()

    # Let this be your custom covariance matrix
    cov_m = customCov()

    # Initialize with custom regularization
    maxdiv = MaxDiversification(reg='entropy', strength=0.01)

    # Optimize portfolio with custom seed and covariance matrix
    weights = maxdiv.optimize(data=training_data, custom_cov=cov_m, seed=46)
    ```

#### `set_regularizer`

```python
def set_regularizer(reg=None, strength=1)
```

Updates the regularization function and its penalty strength.

This method updates both the regularization function and its associated
penalty strength. Useful for changing the behaviour of the optimizer without
initiating a new one.

**Args**

- `reg` (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
- `strength` (*float, optional*): Strength of the regularization. Defaults to `1`.


!!! example "Example:"
    ```python
    # Import the MaxDiversification class
    from opes.objectives.heuristics import MaxDiversification

    # Set with 'entropy' regularization
    optimizer = MaxDiversification(reg='entropy', strength=0.01)

    # --- Do Something with `optimizer` ---
    optimizer.optimize(data=some_data())

    # Change regularizer using `set_regularizer`
    optimizer.set_regularizer(reg='l1', strength=0.02)

    # --- Do something else with new `optimizer` ---
    optimizer.optimize(data=some_data())
    ```

#### `stats`

```python
def stats()
```

Calculates and returns portfolio concentration and diversification statistics.

These statistics help users to inspect portfolio's overall concentration in
allocation. For the method to work, the optimizer must have been initialized, i.e.
the `optimize()` method should have been called at least once for `self.weights`
to be defined other than `None`.

**Returns:**

- A `dict` containing the following keys:
    - `'tickers'` (*list*): A list of tickers used for optimization.
    - `'weights'` (*np.ndarry*): Portfolio weights, output from optimization.
    - `'portfolio_entropy'` (*float*): Shannon entropy computed on portfolio weights.
    - `'herfindahl_index'` (*float*): Herfindahl Index value, computed on portfolio weights.
    - `'gini_coefficient'` (*float*): Gini Coefficient value, computed on portfolio weights.
    - `'absolute_max_weight'` (*float*): Absolute maximum allocation for an asset.

**Raises**

- `PortfolioError`: If weights have not been calculated via optimization.


!!! note "Notes:"
    - All statistics are computed on absolute normalized weights (within the simplex), ensuring compatibility with long-short portfolios.
    - This method is diagnostic only and does not modify portfolio weights.
    - For meaningful interpretation, use these metrics in conjunction with risk and performance measures.

## `REPO`

```python
class REPO(risk_aversion=1, reg=None, strength=1)
```

Return-Entropy Portfolio Optimization (REPO).

Return-Entropy Portfolio Optimization (REPO), introduced by Mercurio et. al.,
applies Shannon entropy as a risk measure for portfolios of continuous-return assets.
The method addresses five key limitations of Markowitz's mean-variance portfolio
optimization: tendency toward sparse solutions with large weights on high-risk assets,
disturbance of asset dependence structures when using investor views, instability of
optimal solutions under input adjustments, difficulty handling non-normal or asymmetric
return distributions, and challenges in estimating covariance matrices. By using entropy
rather than variance as the risk measure, REPO naturally accommodates asymmetric distributions.

**Args**

- `risk_aversion` (*float, optional*): Risk-aversion coefficient. Higher values emphasize risk (entropy) minimization, while lower values favor return seeking. Usually greater than `0`. Defaults to `0.5`.
- `reg` (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
- `strength` (*float, optional*): Strength of the regularization. Defaults to `1`.


### Methods

#### `clean_weights`

```python
def clean_weights(threshold=1e-08)
```

Cleans the portfolio weights by setting very small positions to zero.

Any weight whose absolute value is below the specified `threshold` is replaced with zero.
This helps remove negligible allocations while keeping the array structure intact. This method 
requires portfolio optimization (`optimize()` method) to take place for `self.weights` to be
defined other than `None`.

!!! warning "Warning:"
    This method modifies the existing portfolio weights in place. After cleaning, re-optimization
    is required to recover the original weights.

**Args**

- `threshold` (*float, optional*): Float specifying the minimum absolute weight to retain. Defaults to `1e-8`.


**Returns:**

- `numpy.ndarray`: Cleaned and re-normalized portfolio weight vector.

**Raises**

- `PortfolioError`: If weights have not been calculated via optimization.


!!! note "Notes:"
    - Weights are cleaned using absolute values, making this method compatible with long-short portfolios.
    - Re-normalization ensures the portfolio remains properly scaled after cleaning.
    - Increasing threshold promotes sparsity but may materially alter the portfolio composition.

#### `optimize`

```python
def optimize(
    data=None,
    bin=20,
    custom_mean=None,
    seed=100,
    **kwargs
)
```

Solves the Return-Entropy-Portfolio-Optimization objective:

$$
\min_{\mathbf{w}} \ \gamma \mathcal{H}(\mathbf{r}) - \mathbf{w}^\top \mu + \lambda R(\mathbf{w})
$$

!!! warning "Warning"
    Since REPO objective is generally non-convex, SciPy's `differential_evolution` optimizer
    is used to obtain more robust solutions. This approach incurs significantly higher computational cost and should
    be used with care.

!!! note "Note"
    Asset weight bounds are defaulted to (0,1).

**Args**

- `data` (*pd.DataFrame*): Ticker price data in either multi-index or single-index formats. Examples are given below:
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
  - Open
  - High
  - Low
  - Close
  - Volume
  ```
- `bin` (*int, optional*): Number of histogram bins to be used for the return distribution. Defaults to `20`.
- `custom_mean` (*None or np.ndarray, optional*): Custom mean vector. Can be used to inject externally generated mean vectors (eg. Black-Litterman). Defaults to `None`.
- `seed` (*int or None, optional*): Seed for differential evolution solver. Defaults to `100` to preserve deterministic outputs.
- `**kwargs` (*optional*): Included for interface consistency, allowing the backtesting engine to pass additional or optimizer-specific arguments that may be safely ignored by this optimizer.


**Returns:**

- `np.ndarray`: Vector of optimized portfolio weights.

**Raises**

- `DataError`: For any data mismatch during integrity check.
- `OptimizationError`: If `differential_evolution` solver fails to solve.


!!! example "Example:"
    ```python
    # Importing the REPO module
    from opes.objectives.heuristics import REPO

    # Let this be your ticker data
    training_data = some_data()

    # Let these be your custom mean vector
    mean_v = customMean()

    # Initialize with custom regularization
    repo = REPO(reg='entropy', strength=0.01)

    # Optimize portfolio with custom seed, bins and mean vector
    weights = repo.optimize(data=training_data, bin=15, custom_mean=mean_v, seed=46)
    ```

#### `set_regularizer`

```python
def set_regularizer(reg=None, strength=1)
```

Updates the regularization function and its penalty strength.

This method updates both the regularization function and its associated
penalty strength. Useful for changing the behaviour of the optimizer without
initiating a new one.

**Args**

- `reg` (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
- `strength` (*float, optional*): Strength of the regularization. Defaults to `1`.


!!! example "Example:"
    ```python
    # Import the REPO class
    from opes.objectives.heuristics import REPO

    # Set with 'entropy' regularization
    optimizer = REPO(reg='entropy', strength=0.01)

    # --- Do Something with `optimizer` ---
    optimizer.optimize(data=some_data())

    # Change regularizer using `set_regularizer`
    optimizer.set_regularizer(reg='l1', strength=0.02)

    # --- Do something else with new `optimizer` ---
    optimizer.optimize(data=some_data())
    ```

#### `stats`

```python
def stats()
```

Calculates and returns portfolio concentration and diversification statistics.

These statistics help users to inspect portfolio's overall concentration in
allocation. For the method to work, the optimizer must have been initialized, i.e.
the `optimize()` method should have been called at least once for `self.weights`
to be defined other than `None`.

**Returns:**

- A `dict` containing the following keys:
    - `'tickers'` (*list*): A list of tickers used for optimization.
    - `'weights'` (*np.ndarry*): Portfolio weights, output from optimization.
    - `'portfolio_entropy'` (*float*): Shannon entropy computed on portfolio weights.
    - `'herfindahl_index'` (*float*): Herfindahl Index value, computed on portfolio weights.
    - `'gini_coefficient'` (*float*): Gini Coefficient value, computed on portfolio weights.
    - `'absolute_max_weight'` (*float*): Absolute maximum allocation for an asset.

**Raises**

- `PortfolioError`: If weights have not been calculated via optimization.


!!! note "Notes:"
    - All statistics are computed on absolute normalized weights (within the simplex), ensuring compatibility with long-short portfolios.
    - This method is diagnostic only and does not modify portfolio weights.
    - For meaningful interpretation, use these metrics in conjunction with risk and performance measures.

## `RiskParity`

```python
class RiskParity(reg=None, strength=1)
```

Equal Risk Contribution (Risk Parity) optimization.

Risk parity, developed and popularized by Edward Qian and others in the
1990s-2000s, allocates capital such that each asset contributes equally
to total portfolio risk. The core insight is that market-capitalization
or equal-weight portfolios are dominated by the risk of a few volatile
assets (typically equities), leaving other assets (bonds, commodities)
with minimal risk contribution. Risk parity addresses this by leveraging
low-volatility assets and de-leveraging high-volatility ones to equalize
risk contributions.

**Args**

- `reg` (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
- `strength` (*float, optional*): Strength of the regularization. Defaults to `1`.


### Methods

#### `clean_weights`

```python
def clean_weights(threshold=1e-08)
```

Cleans the portfolio weights by setting very small positions to zero.

Any weight whose absolute value is below the specified `threshold` is replaced with zero.
This helps remove negligible allocations while keeping the array structure intact. This method 
requires portfolio optimization (`optimize()` method) to take place for `self.weights` to be
defined other than `None`.

!!! warning "Warning:"
    This method modifies the existing portfolio weights in place. After cleaning, re-optimization
    is required to recover the original weights.

**Args**

- `threshold` (*float, optional*): Float specifying the minimum absolute weight to retain. Defaults to `1e-8`.


**Returns:**

- `numpy.ndarray`: Cleaned and re-normalized portfolio weight vector.

**Raises**

- `PortfolioError`: If weights have not been calculated via optimization.


!!! note "Notes:"
    - Weights are cleaned using absolute values, making this method compatible with long-short portfolios.
    - Re-normalization ensures the portfolio remains properly scaled after cleaning.
    - Increasing threshold promotes sparsity but may materially alter the portfolio composition.

#### `optimize`

```python
def optimize(
    data=None,
    weight_bounds=(0, 1),
    w=None,
    custom_cov=None
)
```

Solves the Risk Parity objective (Target Contribution Variant):

$$
\min_{\mathbf{w}} \ \sum_i^N \left(RC_i - TC\right)^2
$$

**Args**

- `data` (*pd.DataFrame*): Ticker price data in either multi-index or single-index formats. Examples are given below:
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
  - Open
  - High
  - Low
  - Close
  - Volume
  ```
- `weight_bounds` (*tuple, optional*): Boundary constraints for asset weights. Values must be of the format `(lesser, greater)` with `0 <= |lesser|, |greater| <= 1`. Defaults to `(0,1)`.
- `w` (*None or np.ndarray, optional*): Initial weight vector for warm starts. Mainly used for backtesting and not recommended for user input. Defaults to `None`.
- `custom_cov` (*None or array-like of shape (n_assets, n_assets), optional*): Custom covariance matrix. Can be used to inject externally generated covariance matrices (eg. Ledoit-Wolf). Defaults to `None`.


**Returns:**

- `np.ndarray`: Vector of optimized portfolio weights.

**Raises**

- `DataError`: For any data mismatch during integrity check.
- `OptimizationError`: If `SLSQP` solver fails to solve.


!!! example "Example:"
    ```python
    # Importing the risk parity module
    from opes.objectives.heuristics import RiskParity

    # Let this be your ticker data
    training_data = some_data()

    # Let this be your custom covariance
    cov_m = customCov()

    # Initialize with custom regularization
    rp = RiskParity(reg='entropy', strength=0.01)

    # Optimize portfolio with custom weight bounds and covariance matrix
    weights = rp.optimize(data=training_data, weight_bounds=(0.05, 0.75), custom_cov=cov_m)
    ```

#### `set_regularizer`

```python
def set_regularizer(reg=None, strength=1)
```

Updates the regularization function and its penalty strength.

This method updates both the regularization function and its associated
penalty strength. Useful for changing the behaviour of the optimizer without
initiating a new one.

**Args**

- `reg` (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
- `strength` (*float, optional*): Strength of the regularization. Defaults to `1`.


!!! example "Example:"
    ```python
    # Import the RiskParity class
    from opes.objectives.heuristics import RiskParity

    # Set with 'entropy' regularization
    optimizer = RiskParity(reg='entropy', strength=0.01)

    # --- Do Something with `optimizer` ---
    optimizer.optimize(data=some_data())

    # Change regularizer using `set_regularizer`
    optimizer.set_regularizer(reg='l1', strength=0.02)

    # --- Do something else with new `optimizer` ---
    optimizer.optimize(data=some_data())
    ```

#### `stats`

```python
def stats()
```

Calculates and returns portfolio concentration and diversification statistics.

These statistics help users to inspect portfolio's overall concentration in
allocation. For the method to work, the optimizer must have been initialized, i.e.
the `optimize()` method should have been called at least once for `self.weights`
to be defined other than `None`.

**Returns:**

- A `dict` containing the following keys:
    - `'tickers'` (*list*): A list of tickers used for optimization.
    - `'weights'` (*np.ndarry*): Portfolio weights, output from optimization.
    - `'portfolio_entropy'` (*float*): Shannon entropy computed on portfolio weights.
    - `'herfindahl_index'` (*float*): Herfindahl Index value, computed on portfolio weights.
    - `'gini_coefficient'` (*float*): Gini Coefficient value, computed on portfolio weights.
    - `'absolute_max_weight'` (*float*): Absolute maximum allocation for an asset.

**Raises**

- `PortfolioError`: If weights have not been calculated via optimization.


!!! note "Notes:"
    - All statistics are computed on absolute normalized weights (within the simplex), ensuring compatibility with long-short portfolios.
    - This method is diagnostic only and does not modify portfolio weights.
    - For meaningful interpretation, use these metrics in conjunction with risk and performance measures.

## `SoftmaxMean`

```python
class SoftmaxMean(temperature=1)
```

Softmax Mean portfolio.

The softmax mean portfolio, introduced in recent machine learning-inspired approaches
to portfolio construction, applies the softmax function to expected returns to determine
weights. This method exponentially scales weights based on expected returns through a
temperature parameter $\tau$, providing a smooth interpolation between equal weighting
($\tau \to \infty$) and maximum mean return ($\tau \to 0$). This approach borrows
from the exploration-exploitation tradeoff in reinforcement learning, offering a
principled way to balance return-seeking with diversification.

**Args**

- `temperature` (*float, optional*): Scalar that controls the sensitivity of the weights to return differences. Must be greater than `0`. Defaults to `1.0`.


### Methods

#### `clean_weights`

```python
def clean_weights(threshold=1e-08)
```

Cleans the portfolio weights by setting very small positions to zero.

Any weight whose absolute value is below the specified `threshold` is replaced with zero.
This helps remove negligible allocations while keeping the array structure intact. This method 
requires portfolio optimization (`optimize()` method) to take place for `self.weights` to be
defined other than `None`.

!!! warning "Warning:"
    This method modifies the existing portfolio weights in place. After cleaning, re-optimization
    is required to recover the original weights.

**Args**

- `threshold` (*float, optional*): Float specifying the minimum absolute weight to retain. Defaults to `1e-8`.


**Returns:**

- `numpy.ndarray`: Cleaned and re-normalized portfolio weight vector.

**Raises**

- `PortfolioError`: If weights have not been calculated via optimization.


!!! note "Notes:"
    - Weights are cleaned using absolute values, making this method compatible with long-short portfolios.
    - Re-normalization ensures the portfolio remains properly scaled after cleaning.
    - Increasing threshold promotes sparsity but may materially alter the portfolio composition.

#### `optimize`

```python
def optimize(data=None, custom_mean=None, **kwargs)
```

Satisfies the Softmax Mean objective:

$$
\mathbf{w}_i = \frac{\exp\left( \mu_i / \tau \right)}{\sum_j^N \exp\left( \mu_i / \tau \right)}
$$

!!! note "Note"
    Asset weight bounds are defaulted to (0,1).

**Args**

- `data` (*pd.DataFrame*): Ticker price data in either multi-index or single-index formats. Examples are given below:
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
  - Open
  - High
  - Low
  - Close
  - Volume
  ```
- `custom_mean` (*None or np.ndarray, optional*): Custom mean vector. Can be used to inject externally generated mean vectors (eg. Black-Litterman). Defaults to `None`.
- `**kwargs` (*optional*): Included for interface consistency, allowing the backtesting engine to pass additional or optimizer-specific arguments that may be safely ignored by this optimizer.


**Returns:**

- `np.ndarray`: Vector of optimized portfolio weights.

**Raises**

- `DataError`: For any data mismatch during integrity check.


!!! example "Example:"
    ```python
    # Importing the softmax mean module
    from opes.objectives.heuristics import SoftmaxMean

    # Let this be your ticker data
    training_data = some_data()

    # Let this be your custom mean vector
    # Can be Black-Litterman, Bayesian, Fama-French etc.
    mean_v = customMean()

    # Initialize softmax mean
    soft = SoftmaxMean()

    # Optimize portfolio with custom mean vector
    weights = soft.optimize(data=training_data, custom_mean=mean_v)
    ```

#### `stats`

```python
def stats()
```

Calculates and returns portfolio concentration and diversification statistics.

These statistics help users to inspect portfolio's overall concentration in
allocation. For the method to work, the optimizer must have been initialized, i.e.
the `optimize()` method should have been called at least once for `self.weights`
to be defined other than `None`.

**Returns:**

- A `dict` containing the following keys:
    - `'tickers'` (*list*): A list of tickers used for optimization.
    - `'weights'` (*np.ndarry*): Portfolio weights, output from optimization.
    - `'portfolio_entropy'` (*float*): Shannon entropy computed on portfolio weights.
    - `'herfindahl_index'` (*float*): Herfindahl Index value, computed on portfolio weights.
    - `'gini_coefficient'` (*float*): Gini Coefficient value, computed on portfolio weights.
    - `'absolute_max_weight'` (*float*): Absolute maximum allocation for an asset.

**Raises**

- `PortfolioError`: If weights have not been calculated via optimization.


!!! note "Notes:"
    - All statistics are computed on absolute normalized weights (within the simplex), ensuring compatibility with long-short portfolios.
    - This method is diagnostic only and does not modify portfolio weights.
    - For meaningful interpretation, use these metrics in conjunction with risk and performance measures.

## `Uniform`

```python
class Uniform()
```

Uniform weight portfolio.

The Uniform or 1/N portfolio is the simplest diversification strategy: allocate capital
equally across all available assets. Introduced implicitly throughout financial history
and formalized in academic comparisons by DeMiguel et. al., this approach makes no
assumptions about relative asset qualities. Despite its simplicity, or perhaps because
of it, equal weighting has proven surprisingly difficult to beat out-of-sample. The
strategy completely avoids estimation error in expected returns and correlations,
trading off theoretical optimality for robust performance.

The `Uniform` optimizer does not require any parameters to initialize.

### Methods

#### `clean_weights`

```python
def clean_weights(threshold=1e-08)
```

Cleans the portfolio weights by setting very small positions to zero.

Any weight whose absolute value is below the specified `threshold` is replaced with zero.
This helps remove negligible allocations while keeping the array structure intact. This method 
requires portfolio optimization (`optimize()` method) to take place for `self.weights` to be
defined other than `None`.

!!! warning "Warning:"
    This method modifies the existing portfolio weights in place. After cleaning, re-optimization
    is required to recover the original weights.

**Args**

- `threshold` (*float, optional*): Float specifying the minimum absolute weight to retain. Defaults to `1e-8`.


**Returns:**

- `numpy.ndarray`: Cleaned and re-normalized portfolio weight vector.

**Raises**

- `PortfolioError`: If weights have not been calculated via optimization.


!!! note "Notes:"
    - Weights are cleaned using absolute values, making this method compatible with long-short portfolios.
    - Re-normalization ensures the portfolio remains properly scaled after cleaning.
    - Increasing threshold promotes sparsity but may materially alter the portfolio composition.

#### `optimize`

```python
def optimize(data=None, n_assets=None, **kwargs)
```

Satisfies the 1/N objective:

$$
\mathbf{w}_i = \frac{1}{N} \; \forall \ i=1, ..., N
$$

!!! note "Note"
    Asset weight bounds are defaulted to (0,1).

**Args**

- `data` (*list, pd.DataFrame or None, optional*): List of tickers or ticker price data in either multi-index or single-index formats. Examples are given below:
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
  - Open
  - High
  - Low
  - Close
  - Volume
  ```
- `n_assets` (*int or None, optional*): Number of assets in the portfolio. If this is provided while `data` is `None`, a placeholder ticker list such as `["UNKNOWN", "UNKNOWN", ...]` is automatically generated. Defaults to `None`.
- `**kwargs` (*optional*): Included for interface consistency, allowing the backtesting engine to pass additional or optimizer-specific arguments that may be safely ignored by this optimizer.


**Returns:**

- `np.ndarray`: Vector of equal portfolio weights.

**Raises**

- `DataError`: For any data mismatch during integrity check.


!!! example "Example:"
    ```python
    # Importing the equal-weight module
    from opes.objectives.heuristics import Uniform

    # Let this be your ticker data
    training_data = some_data()

    # Initialize
    equal_weights = Uniform()

    # Optimize portfolio with ticker data
    weights = equal_weights.optimize(data=training_data)

    # Alternatively, optimize for fixed number of assets, here 5
    another_weights = equal_weights.optimize(n_assets=5)
    ```

#### `stats`

```python
def stats()
```

Calculates and returns portfolio concentration and diversification statistics.

These statistics help users to inspect portfolio's overall concentration in
allocation. For the method to work, the optimizer must have been initialized, i.e.
the `optimize()` method should have been called at least once for `self.weights`
to be defined other than `None`.

**Returns:**

- A `dict` containing the following keys:
    - `'tickers'` (*list*): A list of tickers used for optimization.
    - `'weights'` (*np.ndarry*): Portfolio weights, output from optimization.
    - `'portfolio_entropy'` (*float*): Shannon entropy computed on portfolio weights.
    - `'herfindahl_index'` (*float*): Herfindahl Index value, computed on portfolio weights.
    - `'gini_coefficient'` (*float*): Gini Coefficient value, computed on portfolio weights.
    - `'absolute_max_weight'` (*float*): Absolute maximum allocation for an asset.

**Raises**

- `PortfolioError`: If weights have not been calculated via optimization.


!!! note "Notes:"
    - All statistics are computed on absolute normalized weights (within the simplex), ensuring compatibility with long-short portfolios.
    - This method is diagnostic only and does not modify portfolio weights.
    - For meaningful interpretation, use these metrics in conjunction with risk and performance measures.


---

