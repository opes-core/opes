# Distributionally Robust

---

Distributionally robust optimization (DRO) addresses a fundamental challenge in portfolio management: the true distribution of asset returns is unknown, and estimates from historical data are subject to sampling error. Rather than optimizing with respect to a single estimated distributio,n which can lead to severe out-of-sample underperformance when the estimate is poor, DRO optimizes against the worst-case distribution within an ambiguity set of plausible distributions centered around an empirical or reference distribution.

The DRO framework emerged from the recognition that point estimates of return distributions are unreliable, yet we often have confidence that the true distribution lies "close" to our estimate in some sense. By defining an ambiguity set using statistical distance metrics and optimizing for worst-case performance within this set, DRO provides solutions that are robust to estimation error and model misspecification.

The general form of a distributionally robust portfolio optimization problem is:

$$\max_{\mathbf{w}} \min_{\mathbb{P} \in \mathcal{U}} \mathbb{E}_{\mathbb{P}}[u(\mathbf{w}^\top \mathbf{r})]$$

where $\mathbf{w}$ represents portfolio weights, $u(\cdot)$ is the objective function, $\mathbf{r}$ is the random return vector, and $\mathcal{U}$ is the ambiguity set of distributions. The ambiguity set is typically constructed as:

$$\mathcal{U} = \{\mathbb{P} : d(\mathbb{P}, \hat{\mathbb{P}}) \leq \epsilon\}$$

where $\hat{\mathbb{P}}$ is a reference distribution (often the empirical distribution), $d(\cdot, \cdot)$ is a statistical distance metric, and $\epsilon > 0$ controls the size of the ambiguity set. The choice of distance metric fundamentally determines the tractability and conservatism of the resulting problem.

---

## Kullback-Leibler Divergence Ambiguity

The Kullback-Leibler (KL) divergence, introduced by Kullback and Leibler in information theory, measures the relative entropy between two probability distributions. For distributions $\mathbb{P}$ and $\mathbb{Q}$ with densities $p$ and $q$, the KL divergence is:

$$D_{KL}(\mathbb{P} \| \mathbb{Q}) = \int p(x) \log \frac{p(x)}{q(x)} dx$$

The KL-divergence ambiguity set is defined as $\mathcal{U}_{KL}(\epsilon) = \{\mathbb{P} : D_{KL}(\mathbb{P} \| \hat{\mathbb{P}}) \leq \epsilon\}$ where $\epsilon$ controls the radius of uncertainty. A key advantage of KL divergence is that many DRO problems admit tractable convex reformulations through conic duality, as demonstrated by Hu and Hong. The KL framework has strong connections to exponential utility theory and entropic risk measures, making it particularly natural for financial applications.

### Maximum Mean Return

Maximum mean return under KL-divergence uncertainty seeks the portfolio that maximizes expected return in the worst-case distribution within the ambiguity set. This problem was analyzed by Hu and Hong in their comprehensive study of KL-constrained distributionally robust optimization, who showed it admits a tractable convex reformulation through Fenchel duality and change-of-measure techniques.

The distributionally robust maximum mean problem under KL uncertainty can be reformulated as a single-layer convex optimization problem. The worst-case expectation has an analytical form involving the moment generating function of returns, leading to an equivalent deterministic optimization problem that balances expected returns against a robustness penalty.

**Formulation:**

$$\max_{\mathbf{w}} \min_{\mathbb{P} \in \mathcal{U}_{KL}(\epsilon)} \mathbb{E}_{\mathbb{P}}[\mathbf{w}^\top \mathbf{r}]$$

subject to $\mathbf{w}^\top \mathbf{1} = 1$.

This is equivalent to:

$$\min_{\mathbf{w}, \alpha \geq 0} \ \alpha \log \mathbb{E}_{\hat{\mathbb{P}}}[e^{\mathbf{w}^\top \mathbf{r}/\alpha}] + \alpha \epsilon$$

**Usage:**
```python
from opes.objectives.distributionally_robust import KLRobustMaxMean

# Initialize with KL divergence radius
kl_maxmean = KLRobustMaxMean(radius=0.02)
```

### Kelly Criterion

The distributionally robust Kelly criterion addresses estimation error in growth-optimal portfolios by maximizing expected log growth against worst-case distributions within a KL ambiguity set. The KL-robust Kelly criterion produces portfolios that are more diversified than the standard Kelly portfolio, trading off some growth rate for robustness against distributional uncertainty.

**Formulation:**

$$\max_{\mathbf{w}} \min_{\mathbb{P} \in \mathcal{U}_{KL}(\epsilon)} \mathbb{E}_{\mathbb{P}}[\log(1 + \mathbf{w}^\top \mathbf{r})]$$

This is equivalent to:

$$\min_{\mathbf{w}, \alpha \geq 0} \ \alpha \log \mathbb{E}_{\hat{\mathbb{P}}}\left[\left( 1 + f \cdot \mathbf{w}^\top \mathbf{r}  \right)^{-1/\alpha}\right] + \alpha\eta$$

subject to $\mathbf{w}^\top \mathbf{1} = 1$.

**Usage:**
```python
from opes.objectives import KLRobustKelly

# Initialize with KL divergence radius
kl_kelly = KLRobustKelly(radius=0.03)
```

---

## Wasserstein Distance Ambiguity

The Wasserstein distance, rooted in optimal transport theory pioneered by Kantorovich and formalized by Rubinstein, measures the minimum cost of transporting probability mass from one distribution to another. For type-$p$ Wasserstein distance with $p \geq 1$:

$$W_p(\mathbb{P}, \mathbb{Q}) = \left(\inf_{\pi \in \Pi(\mathbb{P}, \mathbb{Q})} \int \|\mathbf{x} - \mathbf{y}\|^p d\pi(\mathbf{x}, \mathbf{y})\right)^{1/p}$$

where $\Pi(\mathbb{P}, \mathbb{Q})$ is the set of joint distributions with marginals $\mathbb{P}$ and $\mathbb{Q}$. The Wasserstein ambiguity set is $\mathcal{U}_W(\epsilon) = \{\mathbb{P} : W_p(\mathbb{P}, \hat{\mathbb{P}}) \leq \epsilon\}$. A key advantage of Wasserstein DRO is that it can handle continuous distributions and provides finite-sample performance guarantees. The type-1 Wasserstein distance admits a particularly tractable dual representation through the Kantorovich-Rubinstein duality theorem, which expresses the distance as a supremum over Lipschitz functions. This duality enables reformulation of Wasserstein DRO problems as finite-dimensional convex programs.

### Maximum Mean Return

Maximum mean return under Wasserstein uncertainty has been studied extensively in the robust optimization literature. The Kantorovich-Rubinstein duality theorem provides an explicit dual reformulation: the worst-case expected return equals the nominal expected return minus a robustness penalty proportional to the maximum expected deviation weighted by asset volatility.

For type-1 Wasserstein distance, the problem admits a particularly simple dual solution. The worst-case distribution shifts probability mass toward lower returns while respecting the Wasserstein budget constraint, with the optimal shift determined by the dual Lipschitz constant.

**Formulation:**

$$\max_{\mathbf{w}} \min_{\mathbb{P} \in \mathcal{U}_{W_1}(\epsilon)} \mathbb{E}_{\mathbb{P}}[\mathbf{w}^\top \mathbf{r}]$$

By Kantorovich-Rubinstein duality, this is equivalent to:

$$
\max_{\mathbf{w}} \mathbb{E}_{\hat{\mathbb{P}}}[\mathbf{w}^\top \mathbf{r}] - \epsilon \|\mathbf{w}\|_*
$$

subject to $\mathbf{w}^\top \mathbf{1} = 1$, where $\|\cdot\|_*$ is the dual norm to the metric used in the Wasserstein distance.

**Usage:**
```python
from opes.objectives.distributionally_robust import WassRobustMaxMean
from opes.objectives.markowitz import MaxMean

# Initialize with Wasserstein radius and ground_norm
wass_maxmean = WassRobustMaxMean(radius=0.01, ground_norm=2)

# For infinite ground norm
wass_maxmean_infinite = WassRobustMaxMean(radius=0.01, ground_norm='inf')

# Manual construction of l-1 ground norm wasserstein
wass_maxmean_manual = MaxMean(reg='l-inf')
```