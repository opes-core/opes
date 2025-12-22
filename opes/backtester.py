import numpy as np
import pandas as pd
import scipy.stats as scistats

from opes.errors import PortfolioError
from opes.utils import slippage, extract_data

class Backtester():
    
    def __init__(self, train_data=None, test_data=None, cost={'const': 10}):
        self.train = train_data
        self.test = test_data
        self.cost = cost
    
    def backtest_integrity_check(self, optimizer, rebalance_freq, seed):
        # Checking optimizer class
        try:
            optimizer.identity
        except:
            raise PortfolioError(f"Portfolio object not given. Got {type(optimizer)}")
        # Checking rebalance frequency type and validity
        if rebalance_freq <= 0 or not isinstance(rebalance_freq, int):
            raise PortfolioError(f"Invalid rebalance frequency. Expected integer within bounds [1,T], Got {rebalance_freq}")
        # Validiating numpy seed
        if seed is not None and not isinstance(seed, int):
            raise PortfolioError(f"Invalid seed. Expected integer or None, Got {seed}")
        # Training data validity
        if len(self.train) <= 5:
            raise PortfolioError(f"Training data insufficient. {len(self.train)} timesteps too short")
        # Cost model validity
        if len(self.cost) != 1:
            raise PortfolioError(f"Invalid cost model. Cost model must be a dictionary of length 1, Got {len(self.cost)}")
        first_key = next(iter(self.cost))
        if first_key.lower() not in ['const', 'lognormal', 'gamma', 'inversegaussian', 'jump']:
            raise PortfolioError(f"Unknown cost model: {first_key}")

    def backtest(self, optimizer, rebalance_freq=None, seed=None):
        
        # Running backtester integrity checks
        self.backtest_integrity_check(optimizer, rebalance_freq, seed)
        # Backtest loop
        test_data = extract_data(self.test)
        # Static weight backtest
        if rebalance_freq is None:
            weights = optimizer.optimize(self.train)
            weights_array = np.tile(weights, (len(test_data), 1))
        # Rolling weight backtest
        if rebalance_freq is not None:
            weights = [None] * len(test_data)
            temp_weights = optimizer.optimize(self.train)
            weights[0] = temp_weights
            for i in range(1, len(test_data)):
                if i % rebalance_freq == 0:
                    # CONFIRM LOOKAHEAD BIAS
                    temp_weights = optimizer.optimize(pd.concat([self.train, self.test.iloc[0:i-1]]).dropna(), w=temp_weights)
                weights[i] = temp_weights
            weights_array = np.vstack(weights)
        portfolio_returns = np.einsum('ij,ij->i', weights_array, test_data)
        portfolio_returns -= slippage(weights=weights_array, returns=portfolio_returns, cost=self.cost, numpy_seed=seed)
        return portfolio_returns

    def get_metrics(self, returns):
        returns = np.array(returns)
        average = returns.mean()
        downside_vol = returns[returns < 0].std()
        vol = returns.std()

        # Performance metrics
        SHARPE = np.sqrt(252) * average / vol if (vol > 0 or not np.isnan(vol)) else np.nan
        SORTINO = np.sqrt(252) * average / downside_vol if (downside_vol > 0 or not np.isnan(downside_vol)) else np.nan
        VOLATILITY = vol * np.sqrt(252) if (vol > 0 or not np.isnan(vol)) else np.nan
        AVERAGE = average
        TOTAL = np.prod(1 + returns) - 1
        CAGR = (1 + TOTAL) ** (252/len(returns)) - 1
        MAX_DD = np.max(1 - np.cumprod(1 + returns) / np.maximum.accumulate(np.cumprod(1 + returns)))
        CALMAR = CAGR / abs(MAX_DD) if MAX_DD > 0 else np.nan
        VAR = -np.quantile(returns, 0.05)
        tail_returns = returns[returns <= -VAR]
        CVAR = -tail_returns.mean() if len(tail_returns) > 0 else np.nan
        SKEW = scistats.skew(returns)
        KURTOSIS = scistats.kurtosis(returns)
        OMEGA = np.sum(np.maximum(returns, 0)) / np.sum(np.maximum(-returns, 0))

        # Zipping Text and values
        performance_metrics = [
            'sharpe',
            'sortino',
            'calmar',
            'volatility',
            'mean_return',
            'total_return',
            'cagr',
            'max_drawdown',
            'var_95',
            'cvar_95',
            'skew',
            'kurtosis',
            'omega_0'
        ]
        values = [VOLATILITY, AVERAGE, TOTAL, CAGR, MAX_DD, VAR, CVAR]
        results = [round(SHARPE, 5), round(SORTINO, 5), round(CALMAR, 5)] + [round(x*100, 5) for x in values] + [round(SKEW, 5), round(KURTOSIS, 5), round(OMEGA, 5)]

        return dict(zip(performance_metrics, results))
