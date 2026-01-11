
---

While utility theory and Markowitz optimization boast heavy theoretical elegance, the quantitative investment world has long recognized that simpler heuristic approaches often outperform their theoretically optimal counterparts in practice. This seemingly paradoxical result stems from the gap between theory and reality: optimal strategies require accurate estimates of expected returns and covariances, yet these parameters are notoriously difficult to estimate from finite, noisy data.

Heuristic portfolios sidestep estimation error by making deliberate simplifying assumptions. Some are practical shortcuts born from trader intuition: equal weighting assets because *"we don't know which will win."* Others are grounded in deeper principles: diversifying risk equally across sources, or minimizing portfolio entropy. What unites these approaches is their robustness: by avoiding strong assumptions about return predictions, they often achieve better out-of-sample performance than optimization-based methods.

---

## Uniform (1/N)

**Description:** The Uniform or 1/N portfolio is the simplest diversification strategy: allocate capital equally across all $N$ available assets. Introduced implicitly throughout financial history and formalized in academic comparisons by DeMiguel et. al., this approach makes no assumptions about relative asset qualities. Despite its simplicity, or perhaps because of it, equal weighting has proven surprisingly difficult to beat out-of-sample. The strategy completely avoids estimation error in expected returns and correlations, trading off theoretical optimality for robust performance. It implicitly assumes all assets have equal expected Sharpe ratios, which while clearly false, may be a better working assumption than noisy sample estimates. Equal weighting also maintains maximum diversification in terms of number of positions and has minimal turnover when rebalancing only to maintain equal weights.

**Formulation:**

$$w_i = \frac{1}{N} \quad \text{for } i = 1, \ldots, N$$

**Usage:**
```python
from opes.methods.heuristics import Uniform

# Initialize equal weight portfolio
ew = Uniform()
```

---

## Risk Parity

**Description:** Risk parity, developed and popularized by Edward Qian and others in the 1990s-2000s, allocates capital such that each asset contributes equally to total portfolio risk. The core insight is that market-capitalization or equal-weight portfolios are dominated by the risk of a few volatile assets (typically equities), leaving other assets (bonds, commodities) with minimal risk contribution. Risk parity addresses this by leveraging low-volatility assets and de-leveraging high-volatility ones to equalize risk contributions. The strategy requires only a covariance matrix estimate, avoiding the treacherous problem of estimating expected returns. Risk parity portfolios tend to be more balanced across asset classes and economic regimes than traditional portfolios, though they may require leverage to achieve target return levels.

**Formulation:**

The target risk contribution for all assets is:

$$TC = \frac{\sqrt{\mathbf{w}^\top\Sigma \mathbf{w}}}{n}$$

with $\sqrt{\mathbf{w}^\top\Sigma \mathbf{w}}$ being the portfolio volatility. Risk parity solves:

$$\min_{\mathbf{w}} \sum_{i=1}^{N} \left(RC_i - TC\right)^2$$

subject to $\mathbf{w}^\top \mathbf{1} = 1$.

**Usage:**
```python
from opes.methods.heuristics import RiskParity

# Initialize risk parity optimizer
rp = RiskParity()
```

---

## Inverse Volatility Portfolio

**Description:** The inverse volatility portfolio, a practical simplification of minimum variance strategies used by practitioners since at least the 1970s, weights assets inversely proportional to their volatilities. The approach is grounded in the intuition that higher-risk assets should receive smaller allocations, which is equivalent to risk parity when all assets are uncorrelated. While deliberately naive about correlations, inverse volatility portfolios are trivial to compute, require minimal data, and often perform surprisingly well out-of-sample.

**Formulation:**

$$w_i = \frac{1/\sigma_i}{\sum_{j=1}^{N} 1/\sigma_j}$$

where $\sigma_i = \sqrt{\Sigma_{ii}}$ is the volatility of asset $i$.

**Usage:**
```python
from opes.methods.heuristics import InverseVolatility

# Initialize inverse volatility portfolio
iv = InverseVolatility()
```

---

## Softmax Mean

**Description:** The softmax mean portfolio, introduced in recent machine learning-inspired approaches to portfolio construction, applies the softmax function to expected returns to determine weights. This method exponentially scales weights based on expected returns through a temperature parameter $\tau$, providing a smooth interpolation between equal weighting ($\tau \to \infty$) and maximum mean return ($\tau \to 0$). This approach borrows from the exploration-exploitation tradeoff in reinforcement learning, offering a principled way to balance return-seeking with diversification.

**Formulation:**

$$w_i = \frac{\exp(\mu_i / \tau)}{\sum_{j=1}^{N} \exp(\mu_j / \tau)}$$

where $\mu_i$ is the expected return of asset $i$ and $\tau > 0$ is the temperature parameter.

**Usage:**
```python
from opes.objectives import SoftmaxMean

# Initialize with temperature parameter
softmax = SoftmaxMean(temperature=1.0)
```

---

## Maximum Diversification

**Description:** The maximum diversification portfolio, introduced by Yves Choueifaty and Yves Coignard, maximizes the diversification ratio: the ratio of weighted-average asset volatility to portfolio volatility. This strategy emerged from the observation that low-volatility portfolios tend to outperform high-volatility ones on a risk-adjusted basis (the low-volatility anomaly). Maximum diversification explicitly seeks portfolios where the diversification benefit is greatest, meaning the portfolio volatility is as far below the weighted average of individual volatilities as possible. The approach requires only the covariance matrix and naturally produces well-diversified portfolios without extreme concentrations. It provides an alternative to minimum variance that maintains exposure to all assets while still emphasizing risk reduction through diversification.

**Formulation:**

$$\max_{\mathbf{w}} \, \frac{\mathbf{w}^\top \boldsymbol{\sigma}}{\sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}}$$

subject to $\mathbf{w}^\top \mathbf{1} = 1$, where $\boldsymbol{\sigma}$ is the vector of asset volatilities.

**Usage:**
```python
from opes.objectives import MaxDiversification

# Initialize maximum diversification optimizer
md = MaximumDiversification()
```

!!! warning "Warning:"
	Since the maximum diversification ratio problem is generally non-convex, we solve it using the `differential_evolution` algorithm from SciPy. As a result, this optimization may exhibit higher computational cost and longer runtimes compared to convex portfolio optimizers.

---

## Return Entropy Portfolio Optimization

**Description:** Return-Entropy Portfolio Optimization (REPO), introduced by Mercurio et. al., applies Shannon entropy as a risk measure for portfolios of continuous-return assets. The method addresses five key limitations of Markowitz's mean-variance portfolio optimization: tendency toward sparse solutions with large weights on high-risk assets, disturbance of asset dependence structures when using investor views, instability of optimal solutions under input adjustments, difficulty handling non-normal or asymmetric return distributions, and challenges in estimating covariance matrices. By using entropy rather than variance as the risk measure, REPO naturally accommodates asymmetric distributions.

**Formulation:**

$$\max_{\mathbf{w}} \, \mathbf{w}^\top \mathbf{\mu} - \gamma \mathcal{H}(\mathbf{r})$$

subject to constraints $\mathbf{w}^\top \mathbf{1} = 1$, where $\mathcal{H}(\mathbf{r})$ is the Shannon entropy of portfolio returns and $\gamma$ is the risk aversion.

**Usage:**
```python
from opes.objectives import REPO

# Initialize return entropy optimizer
re = REPO(risk_aversion=0.5)
```