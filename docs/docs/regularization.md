# Regularization

Regularization schemes penalize undesirable portfolio weight structures during optimization.
They help control concentration, instability, and pathological solutions while trading off
optimality for robustness.

Formally, a regularizer adds a penalty term to the cost objective $\mathcal{L}(\mathbf{w})$ as

$$
\min_{\mathbf{w}} \ \mathcal{L}(\mathbf{w}) + \lambda \cdot R(\mathbf{w})
$$

where $R(\mathbf{w})$ encodes structural preferences over the weights $\mathbf{w}$ and $\lambda$ is the strength of the preference.

---

## Regularization Schemes

| Name         | Formulation                                         | Use-case                                                                                                                                          |
|--------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `l1`       | $\sum_i \lvert \mathbf{w}_i\rvert$                             | Encourages sparse portfolios by driving many weights to zero. Mostly relevant for long-short or unconstrained gross exposure settings.            |
| `l2`       | $\sum_i \mathbf{w}_i^2$                             | Discourages extreme allocations while keeping all assets active. Produces smooth, stable portfolios and reduces sensitivity to noise.             |
| `l-inf`       | $\max_i \lvert \mathbf{w}_i\rvert$                             | Penalizes the largest absolute position, enforcing a soft cap on single-asset dominance. Useful for anti-concentration and anti-YOLO guarantees.  |
| `entropy`  | $-\sum_i \mathbf{w}_i \log \mathbf{w}_i$                     | Encourages diversification by penalizing concentration. Preferred when estimates are noisy and robustness is prioritized over point-optimality.   |
| `variance` | $\ \text{Var}(\mathbf{\mathbf{w}})$                         | Pushes allocations toward uniformity without strictly enforcing equal weights. Useful as a soft equal-weight bias.                                |
| `mpad`     | $\frac{1}{n^2} \sum_{i}^n \sum_{j}^n \lvert \mathbf{w}_i - \mathbf{w}_j\rvert$  | Measures and penalizes inequality across weights. Lower values imply uniform allocations; higher values indicate concentration.                   |

---

## Portfolios without Regularization Support

Not all portfolio objectives in OPES support regularization. Some are inherently robust,
while others rely on non-optimization-based objectives, making regularization inapplicable.
The following objectives do not support regularization:

- `Uniform`
- `InverseVolatility`
- `SoftmaxMean`
- `ExponentialGradient`
- `KLRobustMaxMean`
- `KLRobustKelly`
- `WassRobustMaxMean`

---

## Example

```python

# Importing a valid optimizer from opes
from opes.objectives.markowitz import MaxMean

# Initializing different portfolios with various regularization schemes
maxmean_taxi = MaxMean(reg='l1', strength=0.01)
maxmean_eucl = MaxMean(reg='l2', strength=0.01)
maxmean_cheby = MaxMean(reg='l-inf', strength=0.01)
maxmean_entropy = MaxMean(reg='entropy', strength=0.01)
maxmean_var = MaxMean(reg='variance', strength=0.01)
maxmean_mpad = MaxMean(reg='mpad', strength=0.01)
```




---

