import time
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from opes.methods.risk_metrics import MeanCVaR
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
TICKERS = ["GME", "NVDA", "AAPL", "BRK-B", "SHV"]
train = yf.download(tickers=TICKERS, start="2010-01-01", end="2015-01-01", group_by="ticker", auto_adjust=False)
time.sleep(2) # De-throttling
test = yf.download(tickers=TICKERS, start="2015-01-02", end="2025-01-01", group_by="ticker", auto_adjust=False)

# Ensuring tickers are in same order, yfinance sneaks different orders sometimes
train_tickers = train.columns.get_level_values(0).unique()
test  = test.loc[:, train_tickers]

# Declaring mean-cvar optimizers for various confidence levels
cvar_75 = MeanCVaR(confidence=0.75, risk_aversion=0.5)
cvar_85 = MeanCVaR(confidence=0.85, risk_aversion=0.5)
cvar_95 = MeanCVaR(confidence=0.95, risk_aversion=0.5)
cvar_99 = MeanCVaR(confidence=0.99, risk_aversion=0.5)

# Backtesting strategies
tester = Backtester(train_data=train, test_data=test)
returns75 = tester.backtest(optimizer=cvar_75)
returns85 = tester.backtest(optimizer=cvar_85)
returns95 = tester.backtest(optimizer=cvar_95)
returns99 = tester.backtest(optimizer=cvar_99)

# Plotting wealth
plot_wealth(
    {
        "CVaR 75": returns75,
        "CVaR 85": returns85,
        "CVaR 95": returns95,
        "CVaR 99": returns99
    }
)