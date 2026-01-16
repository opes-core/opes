# `distributionally_robust`

Distributionally robust optimization (DRO) addresses a fundamental challenge in portfolio management:
the true distribution of asset returns is unknown and estimates from historical data are subject to
sampling error. Rather than optimizing with respect to a single estimated distribution which can lead
to severe out-of-sample underperformance when the estimate is poor, DRO optimizes against the worst-case
distribution within an ambiguity set of plausible distributions centered around an empirical or reference
distribution.

OPES leverages dual formulations of selected DRO variants, reducing the resulting problems to tractable,
and in many cases, convex minimization programs. This includes formulations drawn from both established
literature and recent, cutting-edge work, some of which may not yet be fully peer-reviewed. While several
dual representations are well-known (e.g. the Kantorovich-Rubinstein duality), others remain comparatively
less visible in the existing literature.

---





## `KLRobustKelly`

```python
class KLRobustKelly(fraction=1.0, radius=0.01)
```

Kullback-Leibler Ambiguity Kelly Criterion.

Maximizes the expected logarithmic wealth under the worst-case probability
distribution within a specified KL-divergence radius. The distributionally
robust Kelly criterion addresses estimation error in growth-optimal portfolios
by maximizing expected log growth against worst-case distributions within a
KL ambiguity set. The KL-robust Kelly criterion produces portfolios that are
more diversified than the standard Kelly portfolio, trading off some growth
rate for robustness against distributional uncertainty.

**Args**

- `radius` (*float, optional*): The size of the uncertainty set (KL-divergence bound). Larger values indicate higher uncertainty. Defaults to `0.01`.
- `fraction` (*float, optional*): kelly fractional exposure to the market. Must be within (0,1]. Lower values bet less aggressively. Defaults to `1.0`.


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
def optimize(data=None, weight_bounds=(0, 1), w=None)
```

Solves the Hu and Hong KL-Kelly dual objective:

$$
\min_{\mathbf{w}, \alpha \ge 0} \ \alpha \log \mathbb{E}_{\mathbb{P}} \left[\left(1 + f \cdot \mathbf{w}^\top \mathbf{r}\right)^{-1/\alpha}\right] + \alpha \epsilon
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


**Returns:**

- `np.ndarray`: Vector of optimized portfolio weights.

**Raises**

- `DataError`: For any data mismatch during integrity check.
- `PortfolioError`: For any invalid portfolio variable inputs during integrity check.
- `OptimizationError`: If `SLSQP` solver fails to solve.


!!! example "Example:"
    ```python
    # Importing the dro Kelly module
    from opes.objectives.distributionally_robust import KLRobustKelly

    # Let this be your ticker data
    training_data = some_data()

    # Initialize with custom fractional exposure and KL divergence radius
    kl_kelly = KLRobustKelly(fraction=0.8, radius=0.02)

    # Optimize portfolio with custom weight bounds
    weights = kl_kelly.optimize(data=training_data, weight_bounds=(0.05, 0.75))
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

## `KLRobustMaxMean`

```python
class KLRobustMaxMean(radius=0.01)
```

Kullback-Leibler Ambiguity Maximum Mean optimization.

Optimizes the expected return under the worst-case probability distribution
within a KL-divergence uncertainty ball (radius) around the empirical distribution. This problem was
analyzed by Hu and Hong in their comprehensive study of KL-constrained distributionally robust
optimization, who showed it admits a tractable convex reformulation through Fenchel duality
and change-of-measure techniques.

**Args**

- `radius` (*float, optional*): The size of the uncertainty set (KL-divergence bound). Larger values indicate higher uncertainty. Defaults to `0.01`.


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
def optimize(data=None, weight_bounds=(0, 1), w=None)
```

Solves the KL-maximum-mean dual objective:

$$
\min_{\mathbf{w}, \alpha \ge 0} \ \alpha \log \mathbb{E}_{\mathbb{P}} \left[e^{\mathbf{w}^\top \mathbf{r} / \alpha}\right] + \alpha \epsilon
$$

Uses the log-sum-exp technique to solve for numerical stability.

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


**Returns:**

- `np.ndarray`: Vector of optimized portfolio weights.

**Raises**

- `DataError`: For any data mismatch during integrity check.
- `PortfolioError`: For any invalid portfolio variable inputs during integrity check.
- `OptimizationError`: If `SLSQP` solver fails to solve.


!!! example "Example:"
    ```python
    # Importing the dro maximum mean module
    from opes.objectives.distributionally_robust import KLRobustMaxMean

    # Let this be your ticker data
    training_data = some_data()

    # Initialize with KL divergence radius
    kl_maxmean = KLRobustMaxMean(radius=0.02)

    # Optimize portfolio with custom weight bounds
    weights = kl_maxmean.optimize(data=training_data, weight_bounds=(0.05, 0.75))
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

## `WassRobustMaxMean`

```python
class WassRobustMaxMean(radius=0.01, ground_norm=2)
```

Wasserstein Ambiguity Maximum Mean optimization.

Maximum mean return under Wasserstein uncertainty has
been studied extensively in the robust optimization literature. The
Kantorovich-Rubinstein duality theorem provides an explicit dual reformulation.

**Args**

- `radius` (*float, optional*): The size of the uncertainty set (Wasserstein distance bound). Larger values indicate higher uncertainty. Defaults to `0.01`.
- `ground_norm` (*int, optional*): Wasserstein ground norm. Used to find the dual norm for the dual objective. Must be a positive integer. Defaults to `2`.


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
    custom_mean=None
)
```

Solves the Kantorovich-Rubinstein dual objective for type-1 Wasserstein distances:

$$
\min_{\mathbf{w}} \ - \mathbf{w}^\top \mu + \epsilon \| \mathbf{w} \|_{\text{d}}
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
- `custom_mean` (*None or np.ndarray, optional*): Custom mean vector. Can be used to inject externally generated mean vectors (eg. Black-Litterman). Defaults to `None`.


**Returns:**

- `np.ndarray`: Vector of optimized portfolio weights.

**Raises**

- `DataError`: For any data mismatch during integrity check.
- `PortfolioError`: For any invalid portfolio variable inputs during integrity check.
- `OptimizationError`: If `SLSQP` solver fails to solve.


!!! example "Example:"
    ```python
    # Importing the dro maximum mean module
    from opes.objectives.distributionally_robust import WassRobustMaxMean

    # Let this be your ticker data
    training_data = some_data()

    # Let this be your custom mean vector
    mean_v = customMean()

    # Initialize with ground norm and Wasserstein radius
    wass_maxmean = WassRobustMaxMean(radius=0.04, ground_norm=3)

    # Optimize portfolio with custom weight bounds and mean vector
    weights = wass_maxmean.optimize(data=training_data, weight_bounds=(0.05, 0.75), custom_mean=mean_v)
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

## `WassRobustMeanVariance`

```python
class WassRobustMeanVariance(risk_aversion=0.3, radius=0.01, ground_norm=2)
```

Wasserstein Ambiguity Mean-Variance Optimization.

Extends the classical mean-variance framework by incorporating distributional
uncertainty using the Wasserstein distance, following the dual
reformulation approach of Blanchet et al. Instead of optimizing
expected return and variance under a single estimated distribution,
the robust formulation considers the worst-case trade-off over a
Wasserstein ambiguity set around the empirical data. The resulting
dual problem remains tractable and convex, allowing robustness to
be introduced without sacrificing computational feasibility. This
approach mitigates sensitivity to estimation error in both the mean
and covariance and yields portfolios with more reliable out-of-sample
behavior under model misspecification.

**Args**

- `risk_aversion` (*float, optional*): Risk-aversion coefficient. Higher values emphasize risk minimization, while lower values favor return seeking. Usually greater than `0`. Defaults to `0.3`.
- `radius` (*float, optional*): The size of the uncertainty set (Wasserstein distance bound). Larger values indicate higher uncertainty. Defaults to `0.01`.
- `ground_norm` (*int, optional*): Wasserstein ground norm. Used to find the dual norm for the dual objective. Must be a positive integer. Defaults to `2`.


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
    custom_mean=None,
    custom_cov=None
)
```

Solves the Wasserstein Ambiguity Mean-Variance dual objective:

$$
\min_{\mathbf{w}} \ \frac\lambda 2 \left( \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}} + \sqrt{\epsilon} \|\mathbf{w} \|_d \right)^2 - \mathbf{w}^\top \mu + \epsilon \| \mathbf{w} \|_d
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
- `custom_mean` (*None or np.ndarray, optional*): Custom mean vector. Can be used to inject externally generated mean vectors (eg. Black-Litterman). Defaults to `None`.
- `custom_cov` (*None or array-like of shape (n_assets, n_assets), optional*): Custom covariance matrix. Can be used to inject externally generated covariance matrices (eg. Ledoit-Wolf). Defaults to `None`.


**Returns:**

- `np.ndarray`: Vector of optimized portfolio weights.

**Raises**

- `DataError`: For any data mismatch during integrity check.
- `PortfolioError`: For any invalid portfolio variable inputs during integrity check.
- `OptimizationError`: If `SLSQP` solver fails to solve.


!!! example "Example:"
    ```python
    # Importing the dro mean-variance module
    from opes.objectives.distributionally_robust import WassRobustMeanVariance

    # Let this be your ticker data
    training_data = some_data()

    # Let these be your custom mean vector and covariance matrix
    mean_v = customMean()
    cov_m = customCov()

    # Initialize with risk aversion, ground norm and Wasserstein radius
    wass_mean_variance = WassRobustMeanVariance(risk_aversion=0.5, radius=0.04, ground_norm=3)

    # Optimize portfolio with custom weight bounds, mean vector and covariance matrix
    weights = wass_mean_variance.optimize(data=training_data, weight_bounds=(0.05, 0.75), custom_mean=mean_v, custom_cov=cov_m)
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

## `WassRobustMinVariance`

```python
class WassRobustMinVariance(radius=0.01, ground_norm=2)
```

Wasserstein Ambiguity Minimum Variance optimization.

Builds on the distributionally robust
optimization framework developed by Blanchet et al. The method extends the classical
GMV portfolio by minimizing the worst-case portfolio variance over a Wasserstein
ambiguity set centered at the empirical return distribution. Through duality, this
worst-case problem admits a tractable reformulation that preserves convexity while
explicitly controlling sensitivity to distributional misspecification. As a result,
Wasserstein-robust GMV portfolios exhibit improved stability and out-of-sample
performance relative to nominal GMV.

**Args**

- `radius` (*float, optional*): The size of the uncertainty set (Wasserstein distance bound). Larger values indicate higher uncertainty. Defaults to `0.01`.
- `ground_norm` (*int, optional*): Wasserstein ground norm. Used to find the dual norm for the dual objective. Must be a positive integer. Defaults to `2`.


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

Solves the Wasserstein Ambiguity Minimum Variance dual objective:

$$
\min_{\mathbf{w}} \ \left(\sqrt{\mathbf{w}^\top \Sigma \mathbf{w}} + \sqrt{\epsilon} \| \mathbf{w} \|_{\text{d}} \right)^2
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
- `PortfolioError`: For any invalid portfolio variable inputs during integrity check.
- `OptimizationError`: If `SLSQP` solver fails to solve.


!!! example "Example:"
    ```python
    # Importing the dro minimum variance module
    from opes.objectives.distributionally_robust import WassRobustMinVariance

    # Let this be your ticker data
    training_data = some_data()

    # Let this be your custom covariance matrix
    cov_m = customCov()

    # Initialize with ground norm and Wasserstein radius
    wass_minvariance = WassRobustMinVariance(radius=0.04, ground_norm=3)

    # Optimize portfolio with custom weight bounds and covariance matrix
    weights = wass_minvariance.optimize(data=training_data, weight_bounds=(0.05, 0.75), custom_cov=cov_m)
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

