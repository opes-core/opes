# `online`

Online portfolio selection represents a fundamentally different paradigm from
traditional optimization: rather than assuming stationary distributions and
optimizing once, online algorithms sequentially update portfolio weights as new
data arrives, making no statistical assumptions about return distributions. This
framework emerged from machine learning and online learning theory, where the goal
is to compete with the best fixed strategy in hindsight while observing data
only once, in sequence.

!!! warning "Warning:"
    Certain online learning algorithms, currently `ExponentialGradient`,
    only uses the latest return data to update the weights. So, they might work
    suboptimally in backtests having `rebalance_freq` more than `1`.

---





## `BCRP`

```python
class BCRP(reg=None, strength=1)
```

Best Constant Rebalanced Portfolio (BCRP).

Introduced by Thomas Cover in his universal portfolio theory, the BCRP
represents the optimal fixed-weight portfolio that rebalances to constant
proportions after each period. BCRP is the gold standard benchmark in
online portfolio selection: It achieves the maximum wealth that any
constant-proportion strategy could have achieved over the observed sequence.

**Args**

- `reg` (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
- `strength` (*float, optional*): Strength of the regularization. Defaults to `1`.


!!! tip "Tip:"
    Since both Follow-the-Leader (FTL) and Follow-the-Regularized-Leader (FTRL) compute the best constant rebalanced portfolio
    (BCRP) in hindsight to determine the allocation for the subsequent time step, both strategies can be implemented using the
    `BCRP` class.

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
def optimize(data=None, w=None)
```

Solves the BCRP objective:

$$
\min_{\mathbf{w}} \ - \prod^T_t \left(\mathbf{w}^\top \mathbf{x}_t\right) + \lambda R(\mathbf{w})
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
- `w` (*None or np.ndarray, optional*): Initial weight vector for warm starts. Mainly used for backtesting and not recommended for user input. Defaults to `None`.


**Returns:**

- `np.ndarray`: Vector of optimized portfolio weights.

**Raises**

- `DataError`: For any data mismatch during integrity check.
- `PortfolioError`: For any invalid portfolio variable inputs during integrity check.
- `OptimizationError`: If `SLSQP` solver fails to solve.


!!! example "Example:"
    ```python
    # Importing the BCRP module
    from opes.objectives import BCRP

    # Let this be your ticker data
    training_data = some_data()

    # Initialize BCRP for FTL and FTRL respectively
    ftl = BCRP()
    ftrl = BCRP(reg='entropy', strength=0.05)

    # Optimize both FTL and FTRL portfolios
    weights_ftl = ftl.optimize(data=training_data)
    weights_ftrl = ftrl.optimize(data=training_data)
    ```

#### `set_regularizer`

```python
def set_regularizer(reg=None, strength=1)
```

Updates the regularization function and its penalty strength.

This method updates both the regularization function and its associated
penalty strength. It is primarily intended for strategies in which the
regularization must change over time, such as in Follow-the-Regularized-
Leader (FTRL) or other adaptive optimization procedures.

**Args**

- `reg` (*str or None, optional*): Type of regularization to be used. Setting to `None` implies no regularizer. Defaults to `None`.
- `strength` (*float, optional*): Strength of the regularization. Defaults to `1`.


!!! example "Example:"
    ```python
    # Import the BCRP class
    from opes.objectives import BCRP

    # Set with 'entropy' regularization
    ftrl = BCRP(reg='entropy', strength=0.01)

    # --- Do Something with `ftrl` ---
    ftrl.optimize(data=some_data())

    # Change regularizer using `set_regularizer`
    ftrl.set_regularizer(reg='l1', strength=0.02)

    # --- Do something else with new `ftrl` ---
    ftrl.optimize(data=some_data())
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

## `ExponentialGradient`

```python
class ExponentialGradient(learning_rate=0.3)
```

Exponential Gradient (EG) optimizer.

The Exponential Gradient algorithm is a foundational online learning algorithm
that updates portfolio weights using multiplicative updates proportional to exponential returns.
Introduced by Helmbold et. al, it belongs to the family of online convex optimization algorithms
and maintains weights that rise exponentially with cumulative performance.

**Args**

- `learning_rate` (*float, optional*): Learning rate for the EG algorithm. Usually bounded within (0,1]. Defaults to `0.3`.


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
def optimize(data=None, w=None)
```

Performs the Exponential Gradient weight update rule.

$$
\mathbf{w}_{i,t+1} = \mathbf{w}_{i,t} \cdot \exp(\eta \cdot \nabla f_{t,i})
$$

For this implementation, we have taken the reward function $f_{t} = \log \left(1 + \mathbf{w}^\top \mathbf{r}_t\right)$

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
- `w` (*None or np.ndarray, optional*): Previous weight vector for updation. If `None`, previous weights are assumed to be uniform weights. Defaults to `None`.


**Returns:**

- `np.ndarray`: Vector of optimized portfolio weights.

**Raises**

- `DataError`: For any data mismatch during integrity check.
- `PortfolioError`: For any invalid portfolio variable inputs during integrity check.


!!! example "Example:"
    ```python
    # Importing the exponential gradient module
    from opes.objectives import ExponentialGradient as EG

    # Let this be your ticker data
    training_data = some_data()

    # Initialize exponential gradient with high learning rate
    e_g = EG(learning_rate=0.77)

    # Optimize for weights using a previous weight vector
    updated_weights = e_g.optimize(data=training_data, w=prev_weights())
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

## `UniversalPortfolios`

```python
class UniversalPortfolios(
    technique='sample',
    n_samples=100,
    dirichlet_conc=1,
    grid_res=3
)
```

Cover's Universal Portfolio.

Cover's Universal Portfolio is a foundational algorithm in online portfolio
selection that constructs a portfolio as a wealth-weighted average of all
possible constant-rebalanced portfolios over the simplex. Introduced by
Thomas Cover, it belongs to the family of universal portfolio algorithms,
maintaining deterministically updated weights that asymptotically track the
performance of the best constant-rebalanced portfolio in hindsight. The
method leverages a combinatorial integration over the simplex or, in practice,
approximates it via sampling or grid discretization, ensuring parameter-free
portfolio growth in adversarial market environments.

!!! warning "Warning"
    Grid-based Cover's UP scales combinatorially with the number of assets and
    `grid_res`. Large values may cause severe memory and runtime issues. Use with caution.

**Args:**

- `technique` (*str, optional*): Portfolio generation technique. Must be either `"sample"` or `"grid"`. Defaults to `"sample"`.
    - `"sample"`: Generates portfolios stochastically using a Dirichlet distribution.
    - `"grid"`: Generates portfolios deterministically using a simplex grid
- `n_samples` (*int, optional*): Number of portfolios to sample when `technique="sample"`. Ignored when `technique="grid"`. Defaults to `100`.
- `dirichlet_conc` (*float, optional*): Concentration parameter of the Dirichlet distribution used for stochastic portfolio sampling. Defaults to `1`.
- `grid_res` (*int, optional*): Resolution parameter for simplex grid construction when `technique="grid"`. Smaller values are recommended due to combinatorial growth. Defaults to `3`.

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
def optimize(data=None, seed=100, **kwargs)
```

Computes the Cover's Universal Portfolio:

$$
\mathbf{w}_{\text{cover}} = \frac{ \int_\Delta \mathbf{w} \cdot W_T(\mathbf{w}) \ d\mathbf{w}} {\int_\Delta W_T(\mathbf{w}) \ d\mathbf{w}} \approx \frac{ \sum_{\mathbf{w} \in \mathcal P} \mathbf{w} \cdot W_T(\mathbf{w})} {\sum_{\mathbf{w} \in \mathcal P} W_T(\mathbf{w})}
$$

With $\mathcal P \sim \text{Grid}(k)$ or $\mathcal P \sim \text{Dirichlet}(\alpha)$ depending upon user choice.

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
- `seed` (*int or None, optional*): Seed for dirichlet sampling. Defaults to `100` to preserve deterministic outputs.
- `**kwargs` (*optional*): Included for interface consistency, allowing the backtesting engine to pass additional or optimizer-specific arguments that may be safely ignored by this optimizer.


**Returns:**

- `np.ndarray`: Vector of optimized portfolio weights.

**Raises**

- `DataError`: For any data mismatch during integrity check.
- `PortfolioError`: For any invalid portfolio variable inputs during integrity check.


!!! example "Example:"
    ```python
    # Importing the universal portfolios module
    from opes.objectives import UniversalPortfolios as UP

    # Let this be your ticker data
    training_data = some_data()

    # Initialize deterministic Cover's universal portfolio
    deterministic_up = UP(technique='grid', grid_res=5)

    # Initialize stochastic Cover's universal portfolio
    stochastic_up = UP(technique='sample', n_samples=1000, dirichlet_conc=1.5)

    # Optimize both portfolios
    det_up_weights = deterministic_up.optimize(data=training_data)
    stoch_up_weights = stochastic_up.optimize(data=training_data, seed=46)
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

