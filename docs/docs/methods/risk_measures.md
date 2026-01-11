
---

Traditional portfolio optimization, particularly the Markowitz framework, relies on variance as the primary measure of risk. While variance is mathematically convenient, it suffers from fundamental limitations. Variance penalizes upside deviations equally with downside losses, treats all tail events symmetrically, and fails to capture the true nature of catastrophic risk that investors care about most.

Modern risk metrics address these shortcomings by focusing on tail risk: the magnitude and likelihood of large losses. These *coherent risk measures*, formalized by Artzner et. al., satisfy desirable mathematical properties including subadditivity (diversification reduces risk), monotonicity, positive homogeneity and translation invariance. Coherent risk measures provide a principled framework for quantifying downside risk without the pathologies of variance.

The shift from variance to tail risk metrics reflects a deeper understanding of investor preferences and market realities. Real losses are asymmetric and extreme events occur more frequently than normal distributions predict. Fat tails, skewness, and tail dependence dominate portfolio risk during crises, precisely when risk management matters most. Risk metrics that explicitly target tail events align better with investor psychology and regulatory requirements, leading to portfolios that are more robust to extreme market conditions.

!!! note "Author's Recommendation:"
	If you plan on using these portfolios and have low risk assets like ETFs or cash proxies, consider adding a regularizer. Risk measures are very pessimistic and can allocate majority of the capital towards these assets, making them extremely conservative (Yes, EVaR, I'm looking at you).

---

## Conditional Value-at-Risk (CVaR)

**Description:** Conditional Value-at-Risk (CVaR), also known as Expected Shortfall (ES) or Average Value-at-Risk, was introduced by Rockafellar and Uryasev as a coherent alternative to Value-at-Risk (VaR). While VaR asks "what is the maximum loss at the $\alpha$ confidence level?", CVaR asks "what is the expected loss given that we exceed VaR?" This distinction is crucial: VaR ignores the severity of tail losses beyond the threshold, while CVaR captures the full magnitude of catastrophic events.

CVaR is a coherent risk measure, meaning it rewards diversification and satisfies all desirable mathematical properties. It is convex, allowing efficient optimization, and provides stronger tail risk control than VaR. A portfolio optimized for CVaR will not only limit the probability of large losses but also minimize their expected magnitude when they occur. The confidence level $\alpha$ (typically 0.95 or 0.99) determines which tail region to measure: higher $\alpha$ focuses on more extreme events. CVaR has become the standard for risk management in banking regulation (Basel III), insurance and institutional portfolio management due to its theoretical soundness and practical interpretability.

**Formulation:**

$$\text{CVaR}_\alpha(X) = \mathbb{E}[X \mid X \leq \text{VaR}_\alpha(X)]$$

where $X$ is the loss distribution (negative returns) and $\text{VaR}_\alpha(X)$ is the Value-at-Risk at confidence level $\alpha$.

Equivalently, for a portfolio with return $\mathbf{w}^\top \mathbf{r}$:

$$\text{CVaR}_\alpha(\mathbf{w}) = \mathbb{E}[-\mathbf{w}^\top \mathbf{r} \mid -\mathbf{w}^\top \mathbf{r} \geq \text{VaR}_\alpha(\mathbf{w})]$$

**Usage:**
```python
from opes.objectives.risk_metrics import CVaR

# Initialize with confidence level
cvar = CVaR(confidence=0.95)
```

---

## Mean-CVaR Optimization

**Description:** Mean-CVaR optimization, formalized by Rockafellar and Uryasev, combines expected return maximization with CVaR minimization, creating a natural extension of mean-variance optimization that properly accounts for tail risk. The framework trades off expected portfolio return against expected tail losses through a risk aversion parameter, producing a efficient frontier in mean-CVaR space rather than mean-variance space.

This approach addresses the fundamental asymmetry in investor preferences: investors care about expected gains but are particularly averse to large losses. By explicitly targeting tail risk, mean-CVaR portfolios avoid the concentration in fat-tailed assets that mean-variance optimization might favor. Mean-CVaR has become the standard framework for portfolio optimization in contexts where tail risk dominates: hedge funds, pension funds and risk-sensitive institutional investors.

**Formulation:**

$$\max_{\mathbf{w}} \, \mathbb{E}[\mathbf{w}^\top \mathbf{r}] - \gamma \cdot \text{CVaR}_\alpha(\mathbf{w})$$

subject to $\mathbf{w}^\top \mathbf{1} = 1$, where $\gamma > 0$ is the risk aversion parameter.

**Usage:**
```python
from opes.objectives.risk_metrics import MeanCVaR

# Initialize with risk aversion and confidence level
mean_cvar = MeanCVaR(risk_aversion=1.0, confidence=0.95)
```

---

## Entropic Value-at-Risk (EVaR)

**Description:** Entropic Value-at-Risk (EVaR), introduced by Ahmadi-Javid, is a coherent risk measure derived from exponential utility and relative entropy. EVaR emerges naturally when one considers the tightest upper bound on CVaR over all distributions with a given relative entropy from a reference distribution. This connection to information theory provides EVaR with elegant mathematical properties and a principled interpretation.

EVaR is parameterized by a risk aversion coefficient $s > 0$: as $s \to 0$, EVaR converges to the worst-case loss (maximum loss), while as $s \to \infty$, it converges to the expected loss. For intermediate values, EVaR interpolates smoothly between these extremes. EVaR is strictly more sensitive to tail risk than CVaR, making it appropriate for highly risk-averse investors or regulatory contexts requiring conservative risk measurement.

**Formulation:**

$$\text{EVaR}_\alpha(X) = \min_{w, s} \left\{ \frac{1}{s} \ln \left(\frac{\mathbb{E}[e^{sX}]}{1-\alpha}\right) \right\}$$

where $X$ is the loss distribution and $s > 0$ is the risk aversion parameter.

For a portfolio with return $\mathbf{w}^\top \mathbf{r}$:

$$\text{EVaR}_\alpha(\mathbf{w}) = \min_{w, s \gt 0} \left\{ \frac{1}{s} \ln \left(\frac{\mathbb{E}[e^{-s(\mathbf{w}^\top \mathbf{r})}]}{1-\alpha}\right) \right\}$$

**Usage:**
```python
from opes.objectives.risk_metrics import EVaR

# Initialize with tail confidence
evar = EVaR(confidence=0.8)
```

!!! note "Note:"
	In OPES, EVaR’s risk-aversion parameter, $s$,  is not fixed a priori, but is optimized jointly with the portfolio’s loss distribution. This removes arbitrary tuning and yields a coherent, scale-consistent portfolio that is less susceptible to extreme losses and drawdowns for a given confidence level $\alpha$.

## Mean-EVaR Optimization

**Description:** Mean-EVaR optimization extends the mean-CVaR framework by replacing CVaR with the more tail-sensitive EVaR measure. This creates an optimization problem that balances expected return against entropic tail risk, providing stronger protection against extreme losses than mean-CVaR while maintaining computational tractability through convex optimization.

The mean-EVaR framework is particularly valuable for portfolios where estimation error in the tail distribution could lead to catastrophic losses, or where regulatory requirements demand conservative risk quantification. The EVaR component's exponential weighting of tail events means the optimizer naturally avoids positions that could generate extreme losses, even if they appear attractive on a mean-CVaR basis. This two-parameter structure provides fine-grained control over portfolio conservatism.

**Formulation:**

$$\max_{\mathbf{w}} \, \mathbb{E}[\mathbf{w}^\top \mathbf{r}] - \gamma \cdot \text{EVaR}_\alpha(\mathbf{w})$$

subject to $\mathbf{w}^\top \mathbf{1} = 1$, where $\gamma > 0$ is the risk aversion parameter.

**Usage:**
```python
from opes.objectives.risk_metrics import MeanEVaR

# Initialize with risk aversion and tail confidence
mean_evar = MeanEVaR(risk_aversion=1.0, confidence=0.8)
```

## Entropic Risk Measure (ERM)

**Description:** The Entropic Risk Measure (ERM), introduced by Föllmer and Schied in their comprehensive theory of convex risk measures, provides a risk quantification directly derived from exponential utility maximization. ERM is defined as the negative of the exponential certainty equivalent, representing the minimal amount an investor with exponential utility and risk aversion $\gamma$ would accept in place of a random payoff.

ERM has deep connections to information theory and robust optimization. It can be interpreted as the worst-case expected loss under a distorted probability measure, where the distortion is constrained by relative entropy, effectively modeling model uncertainty through probability tilts. This makes ERM naturally suited for robust portfolio optimization where the investor seeks protection against distributional ambiguity. The risk aversion parameter $\gamma$ controls sensitivity: small $\gamma$ produces risk measures close to expected value, while $\gamma \to \infty$ approaches worst-case (maximum loss) sensitivity.

**Formulation:**

$$\text{ERM}_\gamma(X) = \frac{1}{\gamma} \ln \mathbb{E}[e^{\gamma X}]$$

where $X$ is the loss distribution and $\gamma > 0$ is the risk aversion parameter.

For a portfolio with return $\mathbf{w}^\top \mathbf{r}$:

$$\text{ERM}_\gamma(\mathbf{w}) = \frac{1}{\gamma} \ln \mathbb{E}[e^{-\gamma \mathbf{w}^\top \mathbf{r}}]$$

**Usage:**
```python
from opes.objectives import EntropicRisk

# Initialize with risk aversion parameter
erm = EntropicRisk(risk_aversion=2.0)
```

!!! note "Note:"
	ERM measures risk, so lower values indicate less risk. Minimizing ERM is equivalent to maximizing exponential utility.