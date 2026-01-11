
---

## Definition

Regularization schemes penalize undesirable portfolio weight structures during optimization.
They help control concentration, instability, and pathological solutions while trading off optimality for robustness.

Formally, a regularizer adds a penalty term to the cost objective $\mathcal{L}(w)$ as 
$$
\min_w \;  \mathcal{L}(w) + \lambda \cdot R(w)
$$
where $R(w)$ encodes structural preferences over weights and $\lambda$ is the strength of the preference.

---

## Format

When initializing a portfolio, you can specify a regularizer via the `reg` argument and set its strength with `strength`. Example:

```python
# Import the portfolio class
from opes.methods.utility_theory import CRRA

# Initialize with L-infinity regularizer and a strength of 0.05
CRRA_reg = CRRA(risk_aversion=1.7, reg='l-inf', strength=0.05)
```

**Parameters:**

* `reg : str` Type of regularizer to apply.
* `strength : float` Magnitude of the regularization effect.

---

## Portfolios Without Regularization Support

Not all portfolio objectives in OPES support regularization. Some are inherently robust, while others rely on non-optimization-based objectives, making regularization inapplicable.

The following objectives **do not support regularization**:

* `Uniform`
* `InverseVolatility`
* `SoftmaxMean`
* `ExponentialGradient`
* `KLRobustMaxMean`
* `KLRobustKelly`
* `WassRobustMaxMean`

---

## Regularization Schemes

### L1 (Taxicab)

**Description**

L1 regularization penalizes the absolute sum of weights. It encourages sparse portfolios, often driving many weights exactly to zero.

!!! note "Note: " 
	L1 regularization is primarily relevant in long-short portfolio settings where the gross exposure is not constrained to one, as it promotes sparsity while controlling leverage. Since OPES optimizes portfolios under a unit gross-exposure constraint, L1 regularization is not a primary mechanism in this framework. Nevertheless, it is included for scalability considerations and implementation completeness, as it integrates naturally without requiring structural changes to the optimization code.

**Definition**
$$
R_{L1}(w) = \sum_i |w_i|
$$

**Example Usage**

```python
from opes.methods.markowitz import MaxMean
optimizer = MaxMean(reg="l1", strength=0.01)
```

### L2 (Euclidean)

**Description**

L2 penalizes the squared magnitude of weights. It discourages extreme allocations while keeping all assets active. This reduces sensitivity to noise and produces smooth, stable portfolios.

**Definition**
$$
R_{L2}(w) = \sum_i w_i^2
$$

**Example Usage**

```python
from opes.methods.markowitz import MaxMean
optimizer = MaxMean(reg="l2", strength=0.01)
```

### L$\infty$ (Chebyshev)

**Description**

Also known as maximum weight regularization, this regularizer penalizes the largest absolute position in the portfolio.
It enforces a soft upper bound on single-asset dominance. Good for anti-YOLO guarantees.

**Definition**
$$
R_{L\infty}(w) = \max_i |w_i|
$$

**Example Usage**

```python
from opes.methods.markowitz import MaxMean
optimizer = MaxMean(reg="l-inf", strength=0.01)
```

### Shannon Entropy

**Description**

Entropy regularization encourages diversified weight distributions by penalizing concentration and excessive certainty. It is commonly used when estimates are unreliable and robustness is prioritized over point-optimal solutions.

**Definition**
$$
R_{\text{entropy}}(w) = \sum_i |w_i| \log(|w_i|)
$$

**Example Usage**

```python
from opes.methods.markowitz import MaxMean
optimizer = MaxMean(reg="entropy", strength=0.01)
```

### Weight Variance

**Description**

Penalizes variance across weights, encouraging **even allocation**. It pushes portfolios toward equal-weight without enforcing it strictly.

**Definition**
$$
R_{\text{var}}(w) = \text{Var}(w)
$$

**Example Usage**

```python
from opes.methods.markowitz import MaxMean
optimizer = MaxMean(reg="variance", strength=0.01)
```

### MPAD

**Description**

Mean Pairwise Absolute Deviation (MPAD) measures absolute inequality in the distribution of portfolio weights by averaging the absolute differences between all weight pairs.

Lower values indicate more uniform allocations, while higher values reflect increased concentration.

**Definition**
$$
R_{\text{MPAD}}(w) = \frac{1}{n^2} \sum_{i}^n\sum_{j}^n |w_i - w_j|
$$

**Example Usage**

```python
from opes.methods.markowitz import MaxMean
optimizer = MaxMean(reg="mpad", strength=0.01)
```