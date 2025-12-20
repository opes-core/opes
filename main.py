from opes.methods import markowitz
import yfinance as yf

TICKERS = ["GME", "TSLA", "NVDA", "AAPL", "BRK-B", "SHV", "TLT", "PFE"]
return_data = yf.download(tickers=TICKERS, start="2010-01-01", end="2025-01-01", group_by="ticker", auto_adjust=False)

investments = markowitz.MaxSharpe(reg="l2", strength=1)
weights = investments.optimize(data=return_data)
statistics = investments.stats()
for key in statistics:
    print(f"{key}: {statistics[key]}")