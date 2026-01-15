"""
Regularization schemes penalize undesirable portfolio weight structures during optimization.
They help control concentration, instability, and pathological solutions while trading off
optimality for robustness.

Formally, a regularizer adds a penalty term to the cost objective $\\mathcal{L}(\\mathbf{w})$ as

$$
\\min_{\\mathbf{w}} \\ \\mathcal{L}(\\mathbf{w}) + \\lambda \\cdot R(\\mathbf{w})
$$

where $R(\\mathbf{w})$ encodes structural preferences over the weights $\\mathbf{w}$ and $\\lambda$ is the strength of the preference.

---

### Regularization Schemes

| Name       | Formulation                                                                           | Use-case                                                                                                                                |
|------------|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `l1`       | $\\sum_i \\lvert \\mathbf{w}_i\\rvert$                                                | Encourages sparse portfolios by driving many weights to zero. Mostly relevant for long-short or unconstrained gross exposure settings.  |
| `l2`       | $\\sum_i \\mathbf{w}_i^2$                                                             | Produces smooth, stable portfolios and reduces sensitivity to noise.                                                                    |
| `l-inf`    | $\\max_i \\lvert \\mathbf{w}_i\\rvert$                                                | Penalizes the largest absolute position, enforcing a soft cap on single-asset dominance.                                                |
| `entropy`  | $-\\sum_i \\mathbf{w}_i \\log \\mathbf{w}_i$                                          | Encourages diversification by penalizing concentration.                                                                                 |
| `variance` | $\\ \\text{Var}(\\mathbf{w})$                                                         | Pushes allocations toward uniformity without strictly enforcing equal weights.                                                          |
| `mpad`     | $\\frac{1}{n^2} \\sum_{i}^n \\sum_{j}^n \\lvert \\mathbf{w}_i - \\mathbf{w}_j\\rvert$ | Measures and penalizes inequality across weights.                                                                                       |
| `kld`       | $\\ \\text{D}_{\\text{KL}}(\\mathbf{w} \\| \mathbf{u})$                              | Measures Kullback-Leibler divergence from uniform weights.                                                                              |
| `jsd`      | $\\ \\text{D}_{\\text{JSD}}(\\mathbf{w} \\| \mathbf{u})$                              | Measures Jensen-Shannon divergence from uniform weights.                                                                                |

!!! note "Note"
    For long-short portfolios, mathematically grounded regularizers such as `entropy`, `kld`, and `jsd` first normalize the weights
    and constrain them to the simplex before applying the regularization, ensuring mathematical coherence is not violated.

!!! note "Temporary Note"
    Kullback-Leibler regularization and entropy are the exact same, since KL-divergence's prior distribution is uniform weights. However
    it is included so that it *may* be later updated with custom prior distribution (weights).

---

### Portfolios without Regularization Support

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

### Example

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
maxmean_mpad = MaxMean(reg='kld', strength=0.01)
maxmean_mpad = MaxMean(reg='jsd', strength=0.01)
```

"""

import numpy as np
from opes.errors import PortfolioError

# Small epsilon for numerical stability
_SMALL_EPSILON = 1e-12


# Helper function for entropy regularization
# Returns negative entropy value
def _shannon_entropy(w):

    # Absoluting and Normalizing weights
    w = np.abs(w)
    w = w / w.sum()

    # Computing negative entropy
    neg_entropy = np.sum(w * np.log(w + _SMALL_EPSILON))

    return neg_entropy


# Helper function for Kullback-Leibler regularization
# Returns KL-Divergence value from uniform weights
def _kullback_leibler(w):

    # Absoluting and Normalizing weights
    w = np.abs(w)
    w = w / w.sum()

    # Initiating equal weights
    equal_weight = np.ones(len(w)) / len(w)

    # Computing Kullback-Leibler divergence
    kl_reg = np.sum(w * np.log(w / (equal_weight + _SMALL_EPSILON)))

    return kl_reg


# Helper function for Jensen-Shannon regularization
# Returns JS-Divergence value from uniform weights
def _jensen_shannon(w):

    # Absoluting and Normalizing weights
    w = np.abs(w)
    w = w / w.sum()

    # Initiating equal weights
    equal_weight = np.ones(len(w)) / len(w)

    # Computing the middle distribution
    middle_man = (w + equal_weight) / 2

    # Computing first and second KL terms for JSD
    first_kl = np.sum(w * np.log(w / (middle_man + _SMALL_EPSILON)))
    second_kl = np.sum(
        equal_weight * np.log(equal_weight / (middle_man + _SMALL_EPSILON))
    )

    # Computing Jensen-Shannon divergence
    js_reg = (first_kl + second_kl) / 2

    return js_reg


# Regularizer finding function
# Accepts string and returns a function which can be activated while solving the objective
def _find_regularizer(reg):
    regularizer_mapping = {
        # Quick regularizers
        None: lambda w: 0,
        "l1": lambda w: np.sum(np.abs(w)),
        "l2": lambda w: np.sum(w**2),
        "l-inf": lambda w: np.max(np.abs(w)),
        "variance": lambda w: np.var(w) if len(w) >= 2 else 0,
        "mpad": lambda w: np.mean(np.abs(w[:, None] - w[None, :])),
        # Regularizers using a helper function
        # The function is returned instead of lambda
        "entropy": _shannon_entropy,
        "kld": _kullback_leibler,
        "jsd": _jensen_shannon,
    }

    # Checking regularizer validity
    reg = str(reg).lower() if reg is not None else reg
    if reg in regularizer_mapping:
        # Returning valid regularizer
        return regularizer_mapping[reg]
    else:
        raise PortfolioError(f"Unknown regularizer: {reg}")
