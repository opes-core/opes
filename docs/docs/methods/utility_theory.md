
---

Utility theory provides the mathematical foundation for rational decision-making under uncertainty in portfolio management. The framework originated with Daniel Bernoulli's 1738 solution to the St. Petersburg paradox, where he introduced the concept that investors maximize expected utility rather than expected wealth. This insight revolutionized our understanding of risk preferences and remains central to modern portfolio theory.

The foundational work was formalized by John von Neumann and Oskar Morgenstern (1944) in their expected utility theorem, which established axioms for rational choice under uncertainty. Kenneth Arrow and Gérard Debreu later extended this framework in the 1950s, developing general equilibrium theory and clarifying how risk aversion shapes optimal decision-making. Arrow's seminal work on risk aversion measures provided practical tools for characterizing investor preferences through utility functions.

The general expected utility framework posits that investors choose portfolios to maximize:

$$
\max \; \mathbb{E}[U(W)]
$$

where $U(W)$ is the utility function over terminal wealth $W$. The shape of $U$ encodes risk preferences: concave functions represent risk aversion, with the degree of concavity determining how much expected return an investor demands for bearing risk.

---

## Quadratic Utility

**Description:** Introduced by Harry Markowitz, quadratic utility provides the theoretical justification for mean-variance optimization, as it makes expected utility depend only on the mean and variance of returns. This leads to the elegant result that optimal portfolios lie on the efficient frontier.

**Formulation:**

$$U(W) = W - \frac{\gamma}{2}W^2$$

where $\gamma$ is the risk aversion parameter.

**Usage:**
```python
from opes.objectives.utility_theory import QuadraticUtility

# Initialize portfolio with risk aversion parameter
utility = QuadraticUtility(risk_aversion=0.5)
```

---

## CARA (Constant Absolute Risk Aversion)

**Description:** Introduced by John W. Pratt in his analysis of risk aversion measures, CARA utility exhibits constant absolute risk aversion, meaning the investor's absolute risk tolerance (measured in currency) remains unchanged regardless of wealth level. This property makes CARA utility particularly tractable for problems with normally distributed returns, as optimal decisions become independent of initial wealth. However, this wealth-independence is often viewed as unrealistic since wealthier investors typically take larger absolute positions in risky assets.

**Formulation:**

$$U(W) = -\frac{1}{\alpha} e^{-\alpha W}$$

where $\alpha > 0$ is the absolute risk aversion coefficient.

**Usage:**
```python
from opes.objectives.utility_theory import CARA

# Initialize with absolute risk aversion coefficient
utility = CARA(risk_aversion=0.01)
```

---

## CRRA (Constant Relative Risk Aversion)

**Description:** Introduced by Kenneth Arrow and John W. Pratt, CRRA utility, also known as power utility or isoelastic utility, maintains constant relative risk aversion. This means investors maintain constant portfolio proportions regardless of wealth level, a property consistent with empirical observations of investor behavior. The parameter $\gamma$ represents both risk aversion and the inverse of the elasticity of intertemporal substitution. Empirical estimates typically place $\gamma$ between 1 and 10 for most investors. The logarithmic case ($\gamma = 1$) represents the boundary between decreasing and increasing relative risk aversion.

**Formulation:**

$$U(W) = \frac{W^{1-\gamma}}{1-\gamma} \quad \text{for } \gamma \neq 1$$

$$U(W) = \ln(W) \quad \text{for } \gamma = 1$$

where $\gamma > 0$ is the relative risk aversion coefficient.


**Usage:**
```python
from opes.method.utility_theory import CRRA

# Initialize with relative risk aversion coefficient
utility = CRRA(risk_aversion=2.0)
```

!!! note "Note:" 
	For logarithmic utility (CRRA with $\gamma = 1$), use the [Kelly Criterion](#kelly-criterion-and-fractional-kelly) with $f = 1$ instead.

---

## HARA (Hyperbolic Absolute Risk Aversion)

**Description:** HARA utility, which was developed by various researchers in the 1970s, with significant contributions from Robert C. Merton, is a general class that nests many common utility functions as special cases. The absolute risk aversion is hyperbolic. Setting $b = 0$ yields CRRA utility, while letting $\gamma \to \infty$ with appropriate scaling yields CARA utility. Quadratic utility also emerges as a limiting case. HARA utility is particularly valuable in continuous-time portfolio problems and aggregation theorems, as it preserves tractability while allowing flexible risk preferences. The parameters can be calibrated to match observed portfolio behavior across different wealth levels.

**Formulation:**

$$U(W) = \frac{1-\gamma}{\gamma} \left(\frac{aW}{1-\gamma} + b\right)^{\gamma}$$

where $\gamma \neq 1$ is the risk aversion parameter, and $a, b$ are parameters denoting the scale and shift respectively.


**Usage:**
```python
from opes.objectives.utility_theory import HARA

# Initialize with HARA parameters
utility = HARA(risk_aversion=2.0, scale=1.0, shift=1.2)
```

!!! note "Note:"
	Please do not approximate CRRA or CARA using HARA. Each utility has its own objective to preserve interpretability and numerical behavior.

---

## Kelly Criterion and Fractional Kelly

**Description:** The Kelly criterion (John Larry Kelly Jr, 1956) maximizes the expected geometric growth rate of wealth, which is equivalent to CRRA utility with $\gamma = 1$ (logarithmic utility). It has the remarkable property of maximizing long-run wealth with probability 1 under repeated betting scenarios. The criterion naturally balances risk and return: it never risks ruin, and produces higher median outcomes than any other strategy in the long run.

However, full Kelly betting can experience substantial drawdowns and high volatility in the short to medium term. Fractional Kelly is crucial in practice because it provides a robust approach to handle estimation error and risk management. Using a fraction $f$ (commonly 0.25 to 0.5) reduces volatility quadratically while only reducing growth linearly. This makes fractional Kelly particularly valuable when return estimates are uncertain or when psychological constraints limit tolerance for large drawdowns.

**Formulation:**

The Kelly criterion maximizes the expected logarithmic growth rate:

$$\max_{\mathbf{w}} \, \mathbb{E}[\ln(1 + f \cdot \mathbf{w}^\top \mathbf{r})]$$

where $\mathbf{w}$ is the vector of portfolio weights, $\mathbf{r}$ is the vector of returns and $f \in (0, 1]$ is the Kelly fraction parameter.

**Usage:**
```python
from opes.objectives.utility_theory import Kelly

# Initialize Kelly optimizer with fractional exposure
kelly = Kelly(fraction=0.85)
```

!!! note "Fun Fact:"
	This is my favorite portfolio strategy.