from opes.portfolio import Portfolio
from tabulate import tabulate

TICKERS = ["GME", "TSLA", "NVDA", "AAPL"]
AMOUNT = 10000

# Declaring portfolio object
investments = Portfolio(tickers=TICKERS)

# Training
investments.refresh(path="C:\\Users\\nitin\\Downloads\\test.csv")

# Stats before optimization
print("\nPORTFOLIO DATA BEFORE OPTIMIZING")
print(investments.stats())

# Optimizing Portfolio
investments.optimize(method="kelly", fraction=0.9)

print("\nPORTFOLIO DATA AFTER OPTIMIZATION")
print(investments.stats())

# Backtest
performance = investments.backtest(start_date="2010-01-01", end_date="2020-01-01", cost=20)
print("\nPERFORMANCE METRICS (PORTFOLIO)")
print(tabulate(list(performance[0].items()), headers=["METRIC", "VALUE"], tablefmt="plain"))
print("\nPERFORMANCE METRICS (BENCHMARK)")
print(tabulate(list(performance[1].items()), headers=["METRIC", "VALUE"], tablefmt="plain"))