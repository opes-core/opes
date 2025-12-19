from opes.methods.markowitz import MaxMean
import yfinance as yf

TICKERS = ["GME", "TSLA", "NVDA", "AAPL"]
return_data = yf.download(tickers=TICKERS, start="2020-01-01", end="2024-01-01", group_by="ticker", auto_adjust=False)

investments = MaxMean(reg="l2", strength=0.01)
weights = investments.optimize(data=return_data)

print(f"TICKERS: {investments.tickers}")
print(f"MEAN: {investments.mean}")
print(f"WEIGHTS: {weights}")