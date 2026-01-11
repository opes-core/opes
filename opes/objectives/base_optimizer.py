"""
All OPES optimizer objects expose a common set of methods with a consistent interface.
These methods share identical syntax and can be used uniformly across different portfolio 
constructions, independent of the optimizer's specific type or theoretical foundation.

---

"""

from abc import ABC, abstractmethod
import numpy as np
from opes.errors import PortfolioError


class Optimizer(ABC):
    """
    Abstract base class for all portfolio optimization strategies.

    Defines the standard interface for calculating portfolio weights and
    generating portfolio concentration statistics.
    """
    def __init__(self):
        """
        Initializing `Optimizer` sets `self.weights` and `self.tickers` to `None`.
        They are only updated after `optimize()` method is called.
        """
        self.weights = None
        self.tickers = None

    @abstractmethod
    def optimize(self, data):
        """
        Abstract method to optimize portfolio weights based on the provided data. Parameters 
        and constraints vary for different optimizers.

        **Args:**
        
        - `data` (*pd.DataFrame*): Ticker return data in either multi-index or single-index formats. Examples are given below:
            ```
            # Single-Index Example
            Ticker           TSLA      NVDA       GME        PFE       AAPL  ...
            Date
            2015-01-02  14.620667  0.483011  6.288958  18.688917  24.237551  ...
            2015-01-05  14.006000  0.474853  6.460137  18.587513  23.554741  ...
            2015-01-06  14.085333  0.460456  6.268492  18.742599  23.556952  ...
            2015-01-07  14.063333  0.459257  6.195926  18.999102  23.887287  ...
            2015-01-08  14.041333  0.476533  6.268492  19.386841  24.805082  ...
            ...

            # Multi-Index Example Structure (OHLCV)
            Columns:
            + Ticker (e.g. GME, PFE, AAPL, ...)
              + Open
              + High
              + Low
              + Close
              + Volume
            ```
        - `**kwargs`: Additional keyword arguments specific to the optimizer implementation. Supported parameters and their semantics may vary depending on the selected optimizer.

        ---
        """
        pass

    def stats(self):
        """
        Calculates and returns portfolio concentration and diversification statistics. 
        
        These statistics help users to inspect portfolio's overall concentration in 
        allocation. For the method to work, the optimizer must have been initialized, i.e. 
        the `optimize()` method should have been called at least once for `self.weights`
        to be defined other than `None`. 

        **Returns:**

        - A `dict` containing the following keys:
            - `'tickers'` (*list*): A list of tickers used for optimization.
            - `'weights'` (*np.ndarry): Portfolio weights, output from optimization.
            - `'portfolio_entropy'` (*float*): Shannon entropy computed on portfolio weights.
            - `'herfindahl_index'` (*float*): Herfindahl Index value, computed on portfolio weights.
            - `'gini_coefficient'` (*float*): Gini Coefficient value, computed on portfolio weights.
            - `'absolute_max_weight'` (*float*): Absolute maximum allocation for an asset.
        
        Raises:
            PortfolioError: If weights have not been calculated via optimization.
        
        !!! note "Notes:"
            - All statistics are computed on the absolute value of weights, ensuring compatibility with long-short portfolios.
            - This method is diagnostic only and does not modify portfolio weights.
            - For meaningful interpretation, use these metrics in conjunction with risk and performance measures.

        !!! example "Example:"
            ```python
            # Import a portfolio method from OPES
            from opes.objectives.markowitz import MeanVariance
            from some_random_module import data

            # Initialize and optimize
            mvo = MeanVariance(risk_aversion=0.33, reg="mpad", strength=0.02)
            mvo.optimize(data)

            # Obtaining dictionary and displaying items
            statistics_dictionary = mvo.stats()
            for key, value in statistics_dictionary.items():
                print(f"{key}: {value}")
            ```
        ---
        """
        if self.weights is None:
            raise PortfolioError("Weights not optimized")
        else:
            portfolio_entropy = -np.sum(
                np.abs(self.weights) * np.log(np.abs(self.weights) + 1e-12)
            )
            herfindahl_index = np.sum(self.weights**2)
            gini_coeff = np.mean(
                np.abs(self.weights[:, None] - self.weights[None, :])
            ) / (2 * np.mean(np.abs(self.weights)))
            max_weight = np.max(np.abs(self.weights))
            statistics = {
                "tickers": self.tickers,
                "weights": np.round(self.weights, 5),
                "portfolio_entropy": portfolio_entropy,
                "herfindahl_index": herfindahl_index,
                "gini_coefficient": gini_coeff,
                "absolute_max_weight": max_weight,
            }
            return statistics

    def clean_weights(self, threshold=1e-8):
        """
        Cleans the portfolio weights by setting very small positions to zero.

        Any weight whose absolute value is below the specified `threshold` is replaced with zero.
        This helps remove negligible allocations while keeping the array structure intact. This method
        is primarily useful for statistical portfolios with moderate amount of risk aversion, eg. Mean-Variance.
        This method requires portfolio optimization (`optimize()` method) to take place for `self.weights` to be
        defined other than `None`.

        !!! warning "Warning:"
            This method modifies the existing portfolio weights in place. After cleaning, re-optimization 
            is required to recover the original weights.
        Args:
            threshold (*float, optional*): Float specifying the minimum absolute weight to retain. Defaults to `1e-8`.
        
        **Returns:**
        
        - `numpy.ndarray`: Cleaned and re-normalized portfolio weight vector.
        
        Raises:
            PortfolioError: If weights have not been calculated via optimization.
        
        !!! note "Notes:"
            - Weights are cleaned using absolute values, making this method compatible with long-short portfolios.
            - Re-normalization ensures the portfolio remains properly scaled after cleaning.
            - Increasing threshold promotes sparsity but may materially alter the portfolio composition.
        
        !!! example "Example:"
            ```python
            # Import a portfolio method from OPES
            from opes.objectives.markowitz import MeanVariance
            from some_random_module import data

            # Initialize and optimize
            mvo = MeanVariance(risk_aversion=0.33, reg="mpad", strength=0.02)
            mvo.optimize(data)

            # Mean Variance is infamous for tiny allocations
            # We use `clean_weights` method to filter insignificant weights
            cleaned_weights = mvo.clean_weights(threshold=1e-6) # A higher threshold

            # `clean_weights` modifies weights in place, so the cleaned weights
            # are also accessible directly from the portfolio object
            cleaned_weights_can_also_be_obtained_from = mvo.weights
            ```
        """
        if self.weights is None:
            raise PortfolioError("Weights not optimized")
        else:
            self.weights[np.abs(self.weights) < threshold] = 0
            self.weights /= np.abs(self.weights).sum()
            return self.weights