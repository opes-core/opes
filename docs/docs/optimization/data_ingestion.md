
---

All OPES portfolio optimizers share a common data ingestion pipeline. This ensures consistent behavior across classical, utility-based and robust methods.

This section describes how market data should be structured, how OPES interprets it and what transformations occur before optimization.

---

## Expected Data Format

OPES accepts market data as a pandas DataFrame, either single-index or MultiIndex.

### Single-Index Data

* Must contain closing prices only.
* Index: DatetimeIndex
* Columns: Tickers

**Example:**

```text
Ticker           TSLA      NVDA       GME        PFE       AAPL  ...
Date
2015-01-02  14.620667  0.483011  6.288958  18.688917  24.237551  ...
2015-01-05  14.006000  0.474853  6.460137  18.587513  23.554741  ...
2015-01-06  14.085333  0.460456  6.268492  18.742599  23.556952  ...
2015-01-07  14.063333  0.459257  6.195926  18.999102  23.887287  ...
2015-01-08  14.041333  0.476533  6.268492  19.386841  24.805082  ...
...
```

### Multi-Index Data

* Can contain OHLCV or any other price fields.
* Must have a "Close" column in level 1 (caps sensitive).
* Level 0: Ticker
* Level 1: Price field (must include `"Close"`)

**Example (OHLCV):**

```text
Columns:
  ├── Ticker (e.g. GME, PFE, AAPL, ...)
  │     ├── Open
  │     ├── High
  │     ├── Low
  │     ├── Close
  │     └── Volume
```

---

## Missing Data Handling

OPES enforces aligned historical availability across all assets.

!!! danger "Rule: "
    Any row containing at least one NaN across tickers is dropped.

* Assets with shorter trading histories truncate the dataset
* All assets begin optimization from the same effective start date
* Guarantees even sample lengths and proper correlation between returns

---

## Return Construction

After cleaning/truncation, close prices are extracted per asset. Returns are computed as:

$$
R_t = \frac{P_{\text{close}}^{(t)}}{P_{\text{close}}^{(t-1)}} - 1
$$