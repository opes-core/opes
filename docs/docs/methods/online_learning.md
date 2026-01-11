
---

Online portfolio selection represents a fundamentally different paradigm from traditional optimization: rather than assuming stationary distributions and optimizing once, online algorithms sequentially update portfolio weights as new data arrives, making no statistical assumptions about return distributions. This framework emerged from machine learning and online learning theory, where the goal is to compete with the best fixed strategy in hindsight while observing data only once, in sequence.

The online learning perspective is particularly relevant for non-stationary markets where distributions shift over time. These algorithms adapt continuously, treating portfolio selection as a sequential decision problem where each period's returns inform the next period's allocation. While they may seem conservative compared to methods that exploit statistical structure, online algorithms provide a safety net: they guarantee reasonable performance regardless of market behavior, a valuable property when parametric assumptions fail.

!!! warning "Warning:"
	Certain online learning algorithms, currently `ExponentialGradient`, only uses the latest return data to update the weights. So, they might work suboptimally in backtests having a rebalance frequency more than 1. See [Portfolio Backtesting](../backtesting/portfolio_backtesting.md) for more details.

---

## Best Constant Rebalanced Portfolio (BCRP)

**Description:** The Best Constant Rebalanced Portfolio (BCRP), introduced by Thomas Cover in his universal portfolio theory, represents the optimal fixed-weight portfolio that rebalances to constant proportions after each period. BCRP is the gold standard benchmark in online portfolio selection: It achieves the maximum wealth that any constant-proportion strategy could have achieved over the observed sequence. However, BCRP requires complete knowledge of all future returns, making it unrealizable in practice.

The importance of BCRP lies in its role as the comparison point for online algorithms. Any online strategy's regret is measured against BCRP's wealth. Cover proved that the universal portfolio algorithm achieves wealth within a logarithmic factor of BCRP without any statistical assumptions, establishing that Follow The Leader (FTL) and Follow The Regularized Leader (FTRL) strategies can approach BCRP performance. FTL simply chooses each period the portfolio that performed best on all previous data, while FTRL adds [regularization](../regularization.md) to prevent overfitting to noise. BCRP represents the fundamental limit of what can be achieved by constant rebalancing, and the goal of online algorithms is to track it as closely as possible using only past information.

**Formulation:**

BCRP finds the constant portfolio weights $\mathbf{w}^*$ that maximize cumulative wealth:

$$\mathbf{w}^* = \arg\max_{\mathbf{w}} \prod_{t=1}^{T} (\mathbf{w}^\top \mathbf{x}_t)$$

where $\mathbf{x}_t$ is the vector of price relatives (gross returns) at time $t$, and $|\mathbf{w}|^\top \mathbf{1} = 1$.

**Usage:**
```python
from opes.objectives.online import BCRP

# Initialize BCRP (for FTL)
ftl = BCRP()

# Initialize BCRP with entropy regularization (for FTRL)
ftrl = BCRP(reg='entropy', strength=0.01)
```

---

## Exponential Gradient (EG)

**Description:** The Exponential Gradient algorithm, introduced by Helmbold et. al., is a foundational online learning algorithm that updates portfolio weights using multiplicative updates proportional to exponential returns. EG belongs to the family of online convex optimization algorithms and maintains weights that rise exponentially with cumulative performance, that is, assets that have performed well receive exponentially larger allocations.

**Formulation:**

At time $t+1$, update weights using:

$$\tilde{\mathbf{w}}_{i,t+1} = \mathbf{w}_{i,t} \cdot \exp(\eta \cdot \mathbf{r}_{i,t})$$

then normalize:

$$\mathbf{w}_{i,t+1} = \frac{\tilde{\mathbf{w}}_{i,t+1}}{\sum_{j} \tilde{\mathbf{w}}_{j,t+1}}$$

where $\mathbf{w}_{i,t}$ is the weight of asset $i$ at time $t$, $\mathbf{r}_{i,t}$ is the return of asset $i$ at time $t$, and $\eta > 0$ is the learning rate.

**Usage:**
```python
from opes.objectives.online import ExponentialGradient

# Initialize with learning rate
eg = ExponentialGradient(learning_rate=0.02)
```