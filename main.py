from opes.methods import markowitz, utility_theory as eut, risk_metrics as rm, heuristics as h
import yfinance as yf
import numpy as np

TICKERS = ["GME", "TSLA", "NVDA", "AAPL", "BRK-B", "PFE", "SHV"]
return_data = yf.download(tickers=TICKERS, start="2020-01-01", end="2025-01-01", group_by="ticker", auto_adjust=False)

optimizer = h.RiskParity()
weights = optimizer.optimize(data=return_data)
statistics = optimizer.stats()
for key in statistics:
    print(f"{key}: {round(statistics[key], 5) if type(statistics[key]) == np.float64 else statistics[key]}")