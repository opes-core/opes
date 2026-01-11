
---

Markowitz Portfolio Theory (MPT), introduced by Harry Markowitz in his groundbreaking 1952 paper "Portfolio Selection," revolutionized investment management by providing a rigorous mathematical framework for portfolio construction. Prior to Markowitz, investors focused primarily on selecting individual securities with the highest expected returns. Markowitz demonstrated that rational investors should consider the entire portfolio's risk-return profile, not just individual assets in isolation.

The key insight is that diversification reduces risk because asset returns are not perfectly correlated. By combining assets with low or negative correlations, investors can construct portfolios that offer higher expected returns for a given level of risk, or lower risk for a given expected return. This fundamental principle earned Markowitz the Nobel Prize in Economics in 1990.

The Markowitz framework models expected portfolio return as:

$$\mathbb{E}[r_p] = \mathbf{w}^\top \boldsymbol{\mu}$$

and portfolio variance as:

$$\sigma_p^2 = \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}$$

where $\mathbf{w}$ is the vector of portfolio weights, $\boldsymbol{\mu}$ is the vector of expected returns and $\boldsymbol{\Sigma}$ is the covariance matrix. The efficient frontier represents the set of portfolios that maximize expected return for each level of risk, or equivalently, minimize risk for each level of expected return.

---

## Maximum Mean Return

**Description:** Maximum mean represents the limiting case of Markowitz's framework where the investor is risk-neutral and cares only about expected return. This portfolio allocates capital entirely to the asset(s) with the highest expected return, ignoring risk considerations. Without short-sale constraints, this strategy places 100% weight in the single asset with maximum expected return. While theoretically simple, this approach is rarely used in practice as it completely ignores diversification benefits and risk management. It serves primarily as a theoretical benchmark representing the extreme high-risk/high-return point on the efficient frontier.

**Formulation:**

$$\max_{\mathbf{w}} \, \mathbf{w}^\top \boldsymbol{\mu}$$

subject to $\mathbf{w}^\top \mathbf{1} = 1$.

**Usage:**
```python
from opes.objectives.markowitz import MaxMean

# Initialize maximum mean optimizer
max_mean = MaxMean()
```

---

## Global Minimum Variance

**Description:** The Global Minimum Variance portfolio represents the lowest-risk portfolio achievable through diversification, regardless of expected returns. This portfolio lies at the leftmost point of the efficient frontier and has the important property that it does not require estimates of expected returns. This makes GMV portfolios more robust to estimation error than other mean-variance strategies, as expected returns are notoriously difficult to estimate accurately, even more so than covariance matrix. The GMV portfolio is particularly popular among practitioners who are skeptical of return forecasts or who seek a purely defensive allocation.

**Formulation:**

$$\min_{\mathbf{w}} \, \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}$$

subject to $\mathbf{w}^\top \mathbf{1} = 1$.

**Usage:**
```python
from opes.objectives.markowitz import MinVariance

# Initialize GMV optimizer
gmv = MinVariance()
```

---

## Mean-Variance

**Description:** Mean-variance optimization balances the tradeoff between expected return and risk (variance) through the risk aversion parameter $\gamma$. Higher values of $\gamma$ indicate greater risk aversion, leading to more conservative portfolios closer to the GMV portfolio and lower values produce more aggressive allocations closer to maximum return. However the framework has two main limitations. First, it is highly sensitive to inputs, particularly expected return estimates, which can lead to extreme and unstable portfolio weights. Practitioners often address this through regularization techniques, robust optimization, or Bayesian approaches. Second, and more fundamentally, mean-variance optimization assumes returns are normally distributed (or that investors have quadratic utility), using only the first two moments. Real asset returns exhibit fat tails, skewness, and other higher-order moments that variance alone cannot capture. This Gaussian assumption means mean-variance portfolios can be severely suboptimal when returns deviate from normality, particularly during market crises when tail risk materializes.

**Formulation:**

$$\max_{\mathbf{w}} \, \mathbf{w}^\top \boldsymbol{\mu} - \gamma \cdot \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}$$

or equivalently:

$$\min_{\mathbf{w}} \, \gamma \cdot \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w} - \mathbf{w}^\top \boldsymbol{\mu}$$

subject to $\mathbf{w}^\top \mathbf{1} = 1$, where $\gamma$ is the risk aversion parameter.

**Usage:**
```python
from opes.objectives.markowitz import MeanVariance

# Initialize with risk aversion parameter
mv = MeanVariance(risk_aversion=1.0)
```

---

## Maximum Sharpe Ratio

**Description:** The maximum Sharpe ratio portfolio, formalized by William F. Sharpe, follows directly from Markowitz's framework. Also known as the tangency portfolio, this portfolio provides the highest risk-adjusted return as measured by excess return per unit of risk. When a risk-free asset is available, this portfolio represents the optimal risky portfolio for all mean-variance investors regardless of their risk aversion. The maximum Sharpe ratio portfolio is widely used in practice as it represents the "best" risk-return tradeoff available. Like mean-variance optimization, it is sensitive to input estimates, particularly expected returns, and small changes in inputs can lead to dramatically different portfolio weights.

**Formulation:**

$$\max_{\mathbf{w}} \, \frac{\mathbf{w}^\top \boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}}$$

subject to $\mathbf{w}^\top \mathbf{1} = 1$, where $r_f$ is the risk-free rate.

**Usage:**
```python
from opes.objectives.markowitz import MaxSharpe

# Initialize with risk-free rate
max_sharpe = MaxSharpe(risk_free=0.02)
```