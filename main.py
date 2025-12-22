import time
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

from opes.methods.utility_theory import CRRA
from opes.methods.online import ExponentialGradient
from opes.backtester import Backtester

# Plot wealth function
def plot_wealth(returns_dict, initial_wealth=1.0):
    if isinstance(returns_dict, np.ndarray):
        returns_dict = {"Strategy": returns_dict}
    plt.figure(figsize=(12, 6))
    for name, returns in returns_dict.items():
        wealth = initial_wealth * np.cumprod(1 + returns)
        plt.plot(wealth, label=name, linewidth=2)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Wealth", fontsize=12)
    plt.title("Portfolio Wealth Over Time", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Obtaining data
TICKERS = ["AAPL", "NVDA", "SHV", "PFE", "TSLA"]
train = yf.download(tickers=TICKERS, start="2010-01-01", end="2015-01-01", group_by="ticker", auto_adjust=False)
time.sleep(2) # De-throttling
test = yf.download(tickers=TICKERS, start="2015-01-02", end="2020-01-01", group_by="ticker", auto_adjust=False)

# Ensuring tickers are in same order, yfinance sneaks different orders sometimes
train_tickers = train.columns.get_level_values(0).unique()
test  = test.loc[:, train_tickers]

# CRRA vs Exponential Gradient
crra_15 = CRRA(risk_aversion=15)
expgrad = ExponentialGradient(learning_rate=0.25)

# Stochastic backtest (Poisson Compound)
tester = Backtester(train_data=train, test_data=test, cost={'jump': (7, 2.3, 0.3)})
scenario_1 = tester.backtest(optimizer=crra_15, rebalance_freq=1)
scenario_2 = tester.backtest(optimizer=expgrad, rebalance_freq=1)

# Plotting wealth
plot_wealth(
    {
        "CRRA": scenario_1,
        "EG": scenario_2,
    }
)

# Portfolio performance metrics demo
print("CRRA PERFORMANCE")
metrics = tester.get_metrics(scenario_1)
for key in metrics:
    print(f"{key}: {metrics[key]}")
print("\nEXPONENTIAL GRADIENT PERFORMANCE")
metrics = tester.get_metrics(scenario_2)
for key in metrics:
    print(f"{key}: {metrics[key]}")