# Installation

---

This page guides you through installing OPES for experimentation & research.

!!! warning "Warning:"
	OPES is currently under development. While it is relatively stable for experimentation, some features may change or break. Use at your own discretion and always verify results when testing.

---

## Procedure

Python 3.10+ is required for `opes` to run (although it *may* work on some lower versions). `opes` is tested upto Python 3.14. To install a stable release of `opes`, `pip` is recommended for convenience.

### Installation

You can install OPES easily via PyPI using `pip`.

```bash
pip install opes
```

This will fetch the latest stable release and all required dependencies. Alternatively, you are also welcome to install the module directly from GitHub.

```bash
git clone https://github.com/opes-core/opes.git
cd opes-main
pip install .
```

You can also install in editable mode if you plan on making any changes to the source code.

```bash 
# After cloning and in the root of the project
pip install -e .
```

---

### Verification

After installation, make sure everything works by opening Python and importing `opes`.

```python
>>> import opes
>>> opes.__version__
>>> '0.10.0' # May not be the current version but you get the idea
```

You can also verify your installation by using `pip`.

```bash
pip show opes
```

If no errors appear, `opes` is ready to use.

---

## Getting Started

`opes` is designed to be minimalistic and easy to use and learn for any user. Here is an example script which implements my favorite portfolio, the Kelly Criterion.

```python
# I recommend you use yfinance for testing.
# However for serious research, using an external, faster API would be more fruitful.
import yfinance as yf

# Importing our Kelly class
from opes.objectives import Kelly

# ---------- FETCHING DATA ----------
# Obtaining ticker data
# Basic yfinance stuff
TICKERS = ["AAPL", "NVDA", "PFE", "TSLA", "BRK-B", "SHV", "TLT"]
asset_data = yf.download(
    tickers=TICKERS, 
    start="2010-01-01", 
    end="2020-01-01", 
    group_by="ticker", 
    auto_adjust=True
)

# ---------- OPES USAGE ----------
# Initialize a Kelly portfolio with fractional exposure and L2 regularization
# Fractional exposure produces less risky weights and L2 regularization contributes in penalizing concentration
kelly_portfolio = Kelly(fraction=0.8, reg="l2", strength=0.001)

# Compute portfolio weights with custom weight bounds
kelly_portfolio.optimize(asset_data, weight_bounds=(0.05, 0.8))

# Clean negligible allocations
cleaned_weights = kelly_portfolio.clean_weights(threshold=1e-6)

# Output the final portfolio weights
print(cleaned_weights)
```

This showcases the simplicity of the module. However there are far more diverse features you can still explore. If you're looking for more examples, preferably some of them with *"context"*, I recommend you check out the [examples](./examples/good_strategy.md) page within the documentation.