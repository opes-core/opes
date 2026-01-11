
---

## Changing Regularizer

The `set_regularizer` method allows users to update the portfolio’s regularization function and adjust its penalty strength. Regularization helps control portfolio risk, prevent overfitting, and enforce constraints on the weight allocations.

### Method Signature

```python
set_regularizer(self, reg=None, strength=1)
```

**Parameters:**

* `reg : str` Type of regularization.
* `strength : float` strength of regularization.

### Usage Example

```python
from opes.methods.markowitz import MaxMean
from some_random_module import return_data

# Declaring optimizer with L2 regularizer
prtflio = MaxMean(reg="l2", strength=0.01)

# Optimizing portfolio with L2
weights_L2 = prtflio.optimize(data=return_data)

# Changing regularizer to entropy
prtflio.set_regularizer(reg="entropy", strength=0.01)

# Optimizing portfolio with entropy
weights_ent = prtflio.optimize(data=return_data)
```

!!! note "Note:"
	- As stated, optimizers that do not support regularization are not eligible for this method.
	- If no regularization is mentioned (`reg=None`), then the existing regularizer (if any) would be removed.

---

## Portfolio Statistics

The `stats` method computes various concentration and diversification metrics for an optimized portfolio. These statistics help assess how evenly capital is allocated across assets and whether the portfolio is overly concentrated.

### Method Signature

```python
stats(self)
```

### Returns

`dict`
A dictionary containing asset identifiers, optimized weights, and concentration statistics.

**Returned Fields**

| Key                   | Description                                                               |
| --------------------- | ------------------------------------------------------------------------- |
| `Tickers`             | List of asset identifiers in the portfolio                                |
| `Weights`             | Optimized portfolio weights (rounded to 5 decimals)                       |
| `Portfolio Entropy`   | Measures diversification; higher values indicate more uniform allocations |
| `Herfindahl Index`    | Measures concentration; higher values indicate greater concentration      |
| `Gini Coefficient`    | Measures inequality in weight distribution                                |
| `Absolute Max Weight` | Largest absolute allocation to any single asset                           |

### Raises

| Exception        | Condition                                                                       |
| ---------------- | ------------------------------------------------------------------------------- |
| `PortfolioError` | Raised if portfolio weights have not been computed prior to calling this method |

### Usage Example

```python
# Import a portfolio method from OPES
from opes.methods.markowitz import MeanVariance
from some_random_module import data

# Initialize and optimize
mvo = MeanVariance(risk_aversion=0.33, reg="mpad", strength=0.02)
mvo.optimize(data)

# Obtaining dictionary and displaying items
statistics_dictionary = mvo.stats()
for key, value in statistics_dictionary.items():
    print(f"{key}: {value}")
```

!!! note "Notes:"
    * All statistics are computed on the **absolute value of weights**, ensuring compatibility with long-short portfolios.
    * This method is diagnostic only and does not modify portfolio weights.
    * For meaningful interpretation, use these metrics in conjunction with risk and performance measures.

---

## Cleaning Weights

The `clean_weights` method removes negligible portfolio allocations by setting very small weights to zero and re-normalizing the remaining positions. This is useful for improving interpretability and enforcing sparsity after optimization.

### Method Signature

```python
clean_weights(self, threshold=1e-8)
```

**Parameters**

* `threshold : float` Minimum absolute weight required to retain an allocation. Defaults to `1e-8`.

!!! warning "Warning:"
	This method modifies the existing portfolio weights in place. After cleaning, re-optimization is required to recover the original weights.

### Returns

`numpy.ndarray` The cleaned and re-normalized portfolio weight vector.

### Raises

| Exception        | Condition                                           |
| ---------------- | --------------------------------------------------- |
| `PortfolioError` | Raised if portfolio weights have not been optimized |

### Usage Example

```python
# Import a portfolio method from OPES
from opes.methods.markowitz import MeanVariance
from some_random_module import data

# Initialize and optimize
mvo = MeanVariance(risk_aversion=0.33, reg="mpad", strength=0.02)
mvo.optimize(data)

# Mean Variance is infamous for tiny allocations
# We use `clean_weights` method to filter insignificant weights
cleaned_weights = mvo.clean_weights(threshold=1e-6) # A higher threshold

# `clean_weights` modifies weights in place, so the cleaned weights
# are also accessible directly from the portfolio object
cleaned_weights_can_also_be_obtained_from = mvo.weights
```

!!! notes "Notes:"
	* Weights are cleaned using absolute values, making this method compatible with long-short portfolios.
	* Re-normalization ensures the portfolio remains properly scaled after cleaning.
	* Increasing `threshold` promotes sparsity but may materially alter the portfolio composition.
	* This method modifies weights **in-place**.