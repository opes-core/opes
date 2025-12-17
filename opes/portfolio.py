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
    def __init__(self, tickers=None):
        # Assigning base values
        # "We are all consenting adults here"
        self.tickers = None
        self.train_data = None
        self.weights = np.ones(len(tickers)) / len(tickers)
        self.metadata = {
            "tickers" : None,
            "method" : None,
            "weight" : self.weights
        }
    
    # Refreshing training data
    def refresh(self, train_start=None, train_end=None, path=None):
        if (train_start is None or train_end is None) and path is None:
            logger.error("TRAINING PERIOD NOT PROVIDED")
            raise PortfolioError("Training Period not specified")
        elif path is not None:
            logger.info("READING DATA")
            data = readCSV(path)
            if data.empty:
                logger.error("DATA NON-EXISTENT")
                raise DataError("No data in specified file")
            logger.info("READ SUCCESSFUL")
            tickers = data.columns.get_level_values(0).unique().tolist()
            if set(self.tickers) != set(tickers):
                logger.warning("TICKER SET MISMATCH DETECTED. UPDATING INTERNAL TICKER LIST TO MATCH DATA SOURCE")
                self.tickers = tickers
                self.weights = np.ones(len(self.tickers)) / len(self.tickers)
            self.train_data = trimmer(self.tickers, data)
        else:
            try:
                logger.info("FETCHING TRAINING DATA")
                data = yf.download(tickers=self.tickers, start=train_start, end=train_end, group_by="ticker", auto_adjust=True)

                if data.empty:
                    logger.error("FETCH FAILED")
                    raise DataError("No data returned from yfinance")
                logger.info("FETCH SUCCESSFUL")

                # Updating train data and metadata
                self.train_data = trimmer(self.tickers, data)

            except Exception as e:
                logger.exception("FETCH FAILED")
                raise DataError(f"Failed to fetch data due to the underlying error: {e}")
    
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
            horizon=1, 
            confidence=0.9,
            fraction=1,
            rf=0
        ):
        
        # Resetting weights to prevent false convergence
        tickers_length = len(self.tickers)
        tempweights = np.ones(tickers_length) / tickers_length

        # Checking covariance, mean and training data availability before optimization
        if self.mean is None:
            logger.warning("MEAN RETURN NOT SET. DEFAULTING TO SAMPLE MEAN")
            self.setMean(method="sample")
        if self.covar is None:
            logger.warning("COVARIANCE NOT SET. DEFAULTING TO SAMPLE COVARIANCE")
            self.setCovariance(method="sample")
        if self.train_data is None:
            logger.error("MEAN ESTIMATION FAILED. TRAINING PERIOD NOT PROVIDED")
            raise PortfolioError("Training period not specified. Use refresh(train_start, train_end) method to update training data.")

        # Available Optimizers
        optimizers = {
            "gmv": [models.GMV, [tempweights, self.covar * horizon]],
            "mdp": [models.MDP, [tempweights, self.covar * horizon, self.vols() * np.sqrt(horizon)]],
            "mvo": [models.MVO, [tempweights, self.covar * horizon, risk_aversion, self.mean]],
            "sharpe": [models.Sharpe, [tempweights, self.covar * horizon, self.mean, rf]],
            "riskparity": [models.RP, [tempweights, self.covar * horizon]],
            "cvar": [models.CVaR, [tempweights, confidence, self.train_data]],
            "mcvar": [models.MCVaR, [tempweights, risk_aversion, confidence, self.train_data]],
            "kelly": [models.Kelly, [tempweights, fraction, self.train_data]],
            "erm": [models.ERM, [tempweights, risk_aversion, self.train_data]],
            "crra": [models.CRRA, [tempweights, risk_aversion, self.train_data]],
            "quadutil": [models.quadraticutility, [tempweights, risk_aversion, self.train_data]],
            "evar": [models.EVaR, [tempweights, confidence, self.train_data]],
            "1byn": [models.Equal, [tempweights]]
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
