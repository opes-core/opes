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

 | Regularization Scheme                           | Identifier | Formulation                                                                           |
 | ----------------------------------------------- | ---------- | ------------------------------------------------------------------------------------- |
 | Taxicab Norm                                    | `l1`       | $\sum_i \lvert \mathbf{w}_i\rvert$                                                |
 | Euclidean Norm                                  | `l2`       | $\sum_i \mathbf{w}_i^2$                                                             |
 | Chebyshev Norm                                  | `l-inf`    | $\max_i \lvert \mathbf{w}_i\rvert$                                                |
 | Negative Entropy of Weights                     | `entropy`  | $\sum_i \mathbf{w}_i \log \mathbf{w}_i$                                           |
 | Jensen-Shannon Divergence from Uniform Weights  | `jsd`      | $\text{D}_{\text{JSD}}(\mathbf{w} \| \mathbf{u})$                                |
 | Variance of Weights                             | `variance` | $\text{Var}(\mathbf{w})$                                                            |
 | Mean Pairwise Absolute Deviation                | `mpad`     | $\frac{1}{n^2} \sum_{i}^n \sum_{j}^n \lvert \mathbf{w}_i - \mathbf{w}_j\rvert$ |
 | Maximum Pairwise Deviation                      | `mpd`      | $\max_{i,j} \lvert \mathbf{w}_i - \mathbf{w}_j \rvert$                           |
 | Wasserstein-1 Distance from Uniform Weights     | `wass-1`   | $\text{W}_{1}(\mathbf{w}, \mathbf{u})$                                             |

!!! note "Notes"
    - `l1` regularization is mainly used for long-short portfolios to encourage less extreme
    allocations to meet the net exposure of 1. Using it on long-only portfolios is redundant.
    - For long-short portfolios, mathematically grounded regularizers such as `entropy`, `jsd`
    and `wass-1` first normalize the weights and constrain them to the simplex before applying
    the regularization, ensuring mathematical coherence is not violated.

---

## Portfolios without Regularization Support

Not all portfolio objectives in OPES support regularization. Some are inherently robust,
while others rely on non-optimization-based objectives, making regularization inapplicable.
The following objectives do not support regularization:

- `Uniform`
- `InverseVolatility`
- `SoftmaxMean`
- `HierarchicalRiskParity`
- `UniversalPortfolios`
- `ExponentialGradient`
- `KLRobustMaxMean`
- `KLRobustKelly`
- `WassRobustMaxMean`
- `WassRobustMinVariance`
- `WassRobustMeanVariance`

---

## Example

```python

# Importing an optimizer which supports regularization
from opes.objectives import MaxMean

# Initializing different portfolios with various regularization schemes
maxmean_taxi = MaxMean(reg='l1', strength=0.01)             # Taxicab norm
maxmean_eucl = MaxMean(reg='l2', strength=0.01)             # Euclidean norm
maxmean_cheby = MaxMean(reg='l-inf', strength=0.01)         # Chebyshev norm
maxmean_entropy = MaxMean(reg='entropy', strength=0.01)     # Negative entropy of weights
maxmean_jsd = MaxMean(reg='jsd', strength=0.01)             # Jensen-Shannon divergence
maxmean_var = MaxMean(reg='variance', strength=0.01)        # Variance of weights
maxmean_mpad = MaxMean(reg='mpad', strength=0.01)           # Mean pairwise absolute deviation
maxmean_mpd = MaxMean(reg='mpd', strength=0.01)             # Maximum pairwise deviation
maxmean_wass = MaxMean(reg='wass-1', strength=0.01)         # Wasserstein-1
```




---

