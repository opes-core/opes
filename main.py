from opes.methods import markowitz, utility_theory as ut
import yfinance as yf

TICKERS = ["GME", "TSLA", "NVDA", "AAPL", "BRK-B", "SHV", "TLT", "PFE"]
return_data = yf.download(tickers=TICKERS, start="2020-01-01", end="2025-01-01", group_by="ticker", auto_adjust=False)

optimizer = ut.HARA(risk_aversion=1.1)
weights = optimizer.optimize(data=return_data)
statistics = optimizer.stats()
for key in statistics:
    print(f"{key}: {statistics[key]}")