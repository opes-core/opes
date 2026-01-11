
---

Once your desired portfolio is initialized, the `optimize` method computes the optimal portfolio weights based on the given optimizer, data and constraints. This method is common for all portfolio objectives.

---

## Method Signature

```python
optimize(self, data=None, weight_bounds=(0,1), w=None, ...)
```

**Parameters**

* `data : DataFrame` Input market data for optimization. See [Data Ingestion](./data_ingestion.md) for more details.
* `weight_bounds : tuple(float, float)` Boundary constrains for each asset weight. Defaults to `(0,1)`  (long-only). Leverage is not allowed. Some objectives do not have numerical optimization, hence they **do not** support weight bounds (strictly long-only):
      * `Uniform`
      * `InverseVolatility`
      * `SoftmaxMean`
      * `BCRP`
      * `ExponentialGradient`
      * `MeanEVaR`
      * `SoftmaxMean`
      * `REPO`
!!! note "Note:"
      For long–short weight bounds, the net exposure is constrained to 1.

* `w : numpy.ndarray` Optional warm start weights which are useful for backtesting. Users can provide their own weights, but it is generally not recommended.

**Additional Parameters**

* `custom_mean : numpy.ndarray` Optional custom expected returns vector. Can use Black-Litterman, Fama-French or any other custom expected returns model. The list of optimizers which support this parameter are as follows:
      * `MaxMean`
      * `MeanVariance`
      * `MaxSharpe`
      * `MeanCVaR`
      * `MeanEVaR`
      * `SoftmaxMean`
      * `REPO`
      * `WassRobustMaxMean`
* `custom_cov : numpy.ndarray` : Optional custom covariance matrix vector. Can use Black-Litterman, Fama-French or any other custom expected returns model. The list of optimizers which support this parameter are as follows:
      * `MinVariance`
      * `MeanVariance`
      * `MaxSharpe`
      * `MaxDiversification`
      * `RiskParity`
* `seed : None, int` : Optional seed for non-convex objectives using stochastic optimization algorithms. A fixed seed leads to deterministic outputs. Defaults to `100`. The list of optimizers which uses a stochastic solver are as follows:
      * `MaxDiversification`

---

## Returns

`numpy.ndarray` Optimized portfolio weight vector.

---

## Raises

| Exception           | Condition                               |
| ------------------- | --------------------------------------- |
| `OptimizationError` | Raised if the solver fails to converge. |

---

## Usage Example

```python
from opes.objectives.markowitz import MeanVariance
from some_random_module import customCovariance, customMean
from data_module import data

# Demo custom mean and covariance
mea = customMean()
cov = customCovariance()

# Initialize Portfolio
mvo = MeanVariance(risk_aversion=0.4)

# Optimizing portfolio with custom weight bounds, mean and covariance
opt_weights = mvo.optimize(data, weight_bounds=(-0.2, 1), custom_mean=mea, custom_cov=cov)

# Display weights
print(opt_weights)
```

!!! note "Notes:"
      1. The optimization uses the SLSQP (in some cases, `differential_evolution`) solver from SciPy.
      2. Weight bounds are applied **per asset**, and leverage (sum of absolute weights > 1) is not supported.
      3. The method internally prepares constraints and updates portfolio weights upon success.
      4. Some online methods optimizes/updates the portfolio weights only using the most recent return observations. See [Backtesting](../backtesting/portfolio_backtesting.md) for details.