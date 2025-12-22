import time
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

from opes.methods.utility_theory import CRRA
from opes.backtester import Backtester

# Plot wealth function
def plot_wealth(returns_dict, initial_wealth=1.0):
    if isinstance(returns_dict, np.ndarray):
        returns_dict = {"Strategy": returns_dict}
    plt.figure(figsize=(12, 6))
    for name, returns in returns_dict.items():
        wealth = initial_wealth * np.cumprod(1 + returns)
        plt.plot(wealth, label=name, linewidth=2)
    plt.xlabel("Time Period", fontsize=12)
    plt.ylabel("Wealth", fontsize=12)
    plt.title("Portfolio Wealth Over Time", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Obtaining data
TICKERS = ["GME", "NVDA", "SHV", "PFE", "KO"]
train = yf.download(tickers=TICKERS, start="2010-01-01", end="2015-01-01", group_by="ticker", auto_adjust=False)
time.sleep(2) # De-throttling
test = yf.download(tickers=TICKERS, start="2015-01-02", end="2020-01-01", group_by="ticker", auto_adjust=False)

# Ensuring tickers are in same order, yfinance sneaks different orders sometimes
train_tickers = train.columns.get_level_values(0).unique()
print(train_tickers)
test  = test.loc[:, train_tickers]

# Declaring CRRA optimizer
crra_2 = CRRA(risk_aversion=2)

# Stochastic backtest with 3 scenarios
tester = Backtester(train_data=train, test_data=test, cost={'jump': (7, 2.6, 0.3)})
scenario_1 = tester.backtest(optimizer=crra_2, rebalance_freq=21)
scenario_2 = tester.backtest(optimizer=crra_2, rebalance_freq=21)
scenario_3 = tester.backtest(optimizer=crra_2, rebalance_freq=21)

# Plotting wealth
plot_wealth(
    {
        "CRRA (1)": scenario_1,
        "CRRA (2)": scenario_2,
        "CRRA (3)": scenario_3
    }
)

# Portfolio performance metrics demo
metrics = tester.get_metrics(scenario_2)
for key in metrics:
    print(f"{key}: {metrics[key]}")