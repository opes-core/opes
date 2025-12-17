# Standard library
import logging

# Third party libraries
import numpy as np
import yfinance as yf

# Local librarires
import opes.portfolio_models as models
from opes.utils import trimmer, slippage, metrics, plotter, readCSV
from opes.errors import PortfolioError, OptimizationError, DataError

# Logging system
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Adding logging handler if no handlers exist
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", "%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Portfolio Class
class Portfolio:


    # Caching essential values
    def __init__(self):
        # Assigning base values
        self.metadata = {
            "tickers" : None,
            "weights" : None,
            "method" : None
        }
    
    # Returning stats of the portfolio as a dictionary
    def stats(self):
        entropy = -np.sum(self.weights * np.log(self.weights + 1e-12))
        HI = np.sum(self.weights ** 2)
        active_holdings = np.sum(self.weights > 1e-6)
        maxWeight = np.max(self.weights)
        statistics = {
            "tickers":self.tickers, 
            "weights": np.round(self.weights, 2),
            "Entropy": entropy, 
            "Herfindahl Index": HI,
            "Active Holdings" : active_holdings,
            "Max Weight" : maxWeight
        }
        return statistics

    # Optimize function to optimize using a valid optimizer
    def optimize(
            self, 
            method="gmv", 
            risk_aversion=1.2, 
            confidence=0.9,
            fraction=1,
            rf=0,
            train_data=None,
            train_path=None,
            covariance=None,
            mean=None,
            start_weights=None
        ):
        
        # Checking train_data
        if train_data is None and train_path is None:
            logger.error("OPTIMIZATION FAILED. TRAINING PERIOD NOT PROVIDED")
            raise PortfolioError("Training period not specified")
        
        # Checking training data path
        if train_path is not None:
            logger.info("READING DATA")
            try:
                data = readCSV(train_path)
                if data.empty:
                    logger.error("DATA NON-EXISTENT")
                    raise DataError("No data in specified file")
                logger.info("READ SUCCESSFUL")
                tickers = data.columns.get_level_values(0).unique().tolist()
                self.metadata["tickers"] = tickers
                train_data = trimmer(tickers, data)
            except Exception as e:
                raise DataError("Invalid Training Path") from e
        else:
            train_data = trimmer(data)
        
        # Resetting weights to prevent false convergence
        if start_weights is None:
            tickers_length = len(self.tickers)
            start_weights = np.ones(tickers_length) / tickers_length

        # Available Optimizers
        optimizers = {

            # Markowitz
            "gmv": [models.GMV, [start_weights, train_data, covariance]],
            "mvo": [models.MVO, [start_weights, train_data, covariance, risk_aversion, mean]],
            "sharpe": [models.Sharpe, [start_weights, train_data, covariance, mean, rf]],

            # Heuristics
            "mdp": [models.MDP, [start_weights, train_data, covariance]],
            "riskparity": [models.RP, [start_weights, train_data, covariance, np.sqrt(np.diag(covariance))]],
            "1byn": [models.Equal, [start_weights]],

            # Tail metrics
            "cvar": [models.CVaR, [start_weights, confidence, train_data]],
            "mcvar": [models.MCVaR, [start_weights, risk_aversion, confidence, train_data]],
            "erm": [models.ERM, [start_weights, risk_aversion, train_data]],
            "evar": [models.EVaR, [start_weights, confidence, train_data]],

            # Gambler
            "kelly": [models.Kelly, [start_weights, fraction, train_data]],

            # Utility Functions
            "quadutil": [models.quadraticutility, [start_weights, risk_aversion, train_data]],
            "crra": [models.CRRA, [start_weights, risk_aversion, train_data]]
        }

        # Checking if optimizer is valid
        if method.lower() in optimizers:
            function, args = optimizers[method.lower()]
            self.weights = function(*args)
        else:
            logger.error("UNKNOWN OPTIMIZER")
            raise OptimizationError(f"Unknown Optimizer method: {method}")

    # Portfolio Performance Analysis
    def backtest(self, start_date="2017-01-01", end_date="2018-01-01", cost=0, showfig=True, savefig=False):

        # Initiating Backtest
        try:
            logger.info("INITIATING PORTFOLIO BACKTEST")
            logger.info("FETCHING BACKTEST DATA")
            closes = yf.download(tickers=self.tickers, start=start_date, end=end_date, group_by="ticker", auto_adjust=True)
            
            if closes.empty:
                logger.warning("FETCH FAILED")
                raise DataError("No data returned from yfinance")
            logger.info("FETCH SUCCESSFUL")

            # Truncating to level data
            closes = closes.loc[:, (slice(None), "Close")]
            closes.columns = closes.columns.get_level_values(0)
            df = closes.pct_change(fill_method=None)[self.tickers].dropna()
            equityDates = df.index
            equityReturns = df.values

            # Benchmark Weights
            benchmarkWeights = np.ones(len(self.weights))
            benchmarkWeights /= np.sum(benchmarkWeights)

            # Backtest loop
            portfolioReturns, benchmarkReturns = [], []
            T = len(equityReturns)
            cost /= 10000    # Converting from bps to decimal
            for t in range(T):
                if t != 0:
                    slip = slippage(self.weights, equityReturns[t-1], cost)
                    slipB = slippage(benchmarkWeights, equityReturns[t-1], cost)
                else:
                    slip, slipB = 0, 0

                portfolioReturn = self.weights @ equityReturns[t] - slip
                benchmarkReturn = benchmarkWeights @ equityReturns[t] - slipB
                portfolioReturns.append(portfolioReturn)
                benchmarkReturns.append(benchmarkReturn)
            logger.info("BACKTEST SUCCESSFUL")
            
            # Plotting Wealth Performance
            if savefig or showfig:
                plotter(portfolioReturns, benchmarkReturns, equityDates, showfig, savefig)

            # Returning performance metrics
            return ([metrics(portfolioReturns, T), metrics(benchmarkReturns, T)])
        
        except Exception as e:
            logger.exception("BACKTEST FAILED")
            raise PortfolioError(f"Failed to complete backtest due to the underlying error: {e}")
