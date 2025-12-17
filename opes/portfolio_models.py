from scipy.optimize import minimize
from opes.errors import OptimizationError
import numpy as np
import logging

# Logging system
# Taking config from root
logger = logging.getLogger(__name__)

# Global Minimum Variance Variance model
def GMV(w, SIGMA):

    logger.info("GMV OPTIMIZATION INITIATED")

    # Global Minimum Variance Objective
    # minimize {weights.T * COVARIANCE * weights}
    def f(w):
        return w @ SIGMA @ w
    
    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        logger.info("GMV OPTIMIZATION SUCCESSFUL")
        return result.x
    else:
        logger.error("GMV OPTIMIZATION FAILED")
        raise OptimizationError("Global Minimum Variance Optimization failed")

# Maximum Diversification Portfolio Model
def MDP(w, SIGMA, sigma):

    logger.info("MDP OPTIMIZATION INITIATED")

    # MDP Objecitve
    # maximize {(weights.T * assetVolatility) / sqrt(VARIANCE)}
    def f(w):
        var = w @ SIGMA @ w
        weightvol = w @ sigma

        return -(weightvol / np.sqrt(var))
    
    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        logger.info("MDP OPTIMIZATION SUCCESSFUL")
        return result.x
    else:
        logger.error("MDP OPTIMIZATION FAILED")
        raise OptimizationError("Max Diversification Optimization failed")

# Mean-Variance model
def MVO(w, SIGMA, LAMBDA, MU):

    logger.info("MVO OPTIMIZATION INITIATED")

    # MVO Objective
    # maximize {weights.T * returns - VARIANCE}
    def f(w):
        return (LAMBDA/2)*(w @ SIGMA @ w) - (w @ MU)
    
    # Robustness check
    if LAMBDA < 0:
        logger.error("INVALID RISK AVERSION")
        raise OptimizationError("Risk Aversion is out of bounds [0,inf)")
    
    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        logger.info("MVO OPTIMIZATION SUCCESSFUL")
        return result.x
    else:
        logger.error("MVO OPTIMIZATION FAILED")
        raise OptimizationError("Mean-Variance Optimization failed")

# Maximum Sharpe Model
def Sharpe(w, SIGMA, MU, rf=0):

    logger.info("SHARPE OPTIMIZATION INITIATED")

    # Sharpe Objective
    # maximize (wTr - rf) / Variance^0.5
    def f(w):
        return -((w @ MU) - rf) / max(np.sqrt((w @ SIGMA @ w), 1e-10))
    
    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        logger.info("SHARPE OPTIMIZATION SUCCESSFUL")
        return result.x
    else:
        logger.error("SHARPE OPTIMIZATION FAILED")
        raise OptimizationError("Maximum Sharpe Optimization failed")

# Risk Parity Model
def RP(w, SIGMA):
    
    logger.info("RISK PARITY OPTIMIZATION INITIATED")

    # Risk Parity Objective
    # minimize (RiskContributionOfAsset - targetContribution)^2  (Squared Error)
    def f(w):
        VOL = max(np.sqrt((w @ SIGMA @ w)), 1e-10)
        rc = w * (SIGMA @ w) / VOL
        target = VOL / len(w)
        return np.sum((rc - target)**2)

    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        logger.info("RISK PARITY OPTIMIZATION SUCCESSFUL")
        return result.x
    else:
        logger.error("RISK PARITY OPTIMIZATION FAILED")
        raise OptimizationError("Risk Parity Optimization failed")

# Conditional Value-at-Risk (CVaR) model
def CVaR(w, ALPHA, data):
    
    logger.info("CVAR OPTIMIZATION INITIATED")

    # CVaR Objective 
    # Minimize {v + 1/(1-CONFIDENCE)N * SUM{N} (max(-wR-v, 0))}
    def f(x):
        w = x[:-1]
        v = x[-1]

        X = -data @ w
        excess = np.maximum(X - v, 0.0)

        return v + excess.mean() / (1 - ALPHA)

    # Robustness check
    if ALPHA > 1 or ALPHA < 0:
        logger.error("INVALID CVAR CONFIDENCE")
        raise OptimizationError("Confidence is out of bounds (0,1)")
    
    V = 1
    x0 = np.append(w, V)

    result = minimize(f, x0, method='SLSQP', bounds=[(0,1)]*len(w) + [(None,None)], constraints= [{'type':'eq','fun': lambda x: x[:-1].sum()-1}])
    if result.success:
        logger.info("CVAR OPTIMIZATION SUCCESSFUL")
        return result.x[:-1]
    else:
        logger.error("CVAR OPTIMIZATION FAILED")
        raise OptimizationError("CVaR Optimization failed")

# Mean Conditional Value-at-Risk (MCVaR) model
def MCVaR(w, LAMBDA, ALPHA, data):
    
    logger.info("MCVAR OPTIMIZATION INITIATED")

    # MCVaR Objective 
    # Maximize {weights.T * returns - LAMBDA * CVaR}
    def f(x):
        w = x[:-1]
        v = x[-1]
        mean = np.mean((data @ w))
        X = -data @ w
        excess = np.maximum(X - v, 0.0)

        return LAMBDA * (v + excess.mean() / (1 - ALPHA)) - mean
    
    # Robustness check
    if ALPHA > 1 or ALPHA < 0:
        logger.error("INVALID CVAR CONFIDENCE")
        raise OptimizationError("Confidence is out of bounds (0,1)")
    
    V = 1
    x0 = np.append(w, V)

    result = minimize(f, x0, method='SLSQP', bounds=[(0,1)]*len(w) + [(None), (None)], constraints= [{'type':'eq','fun': lambda x: x[:-1].sum()-1}])
    if result.success:
        logger.info("MCVAR OPTIMIZATION SUCCESSFUL")
        return result.x[:-1]
    else:
        logger.error("MCVAR OPTIMIZATION FAILED")
        raise OptimizationError("Mean-CVaR Optimization failed")

# Kelly Criterion Model
def Kelly(w, fr, data):
    logger.info("KELLY OPTIMIZATION INITIATED")

    # Kelly Objective 
    # Minimize -E[log(1+wTr)]
    def f(w):
        r = fr * np.maximum((data @ w), -0.99)
        return -np.mean(np.log(1 + r))

    # Robustness check
    if fr > 1 or fr <= 0:
        logger.error("INVALID KELLY FRACTION")
        raise OptimizationError("Fraction is out of bounds (0,1]")

    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        logger.info("KELLY OPTIMIZATION SUCCESSFUL")
        return result.x
    else:
        logger.error("KELLY OPTIMIZATION FAILED")
        raise OptimizationError("Kelly Optimization failed")
    
# Constant Relative Risk Aversion Model
def CRRA(w, GAMMA, data):
    logger.info("CRRA OPTIMIZATION INITIATED")

    # CRRA Objective 
    # Minimize -E[W^(1-gamma)/(1-gamma)]
    def f(w):
        X = 1 + np.maximum((data @ w),-0.99)
        return -np.mean(X ** (1-GAMMA)) / (1-GAMMA)

    # Robustness check
    if GAMMA <= 1:
        logger.error("INVALID RISK AVERSION")
        raise OptimizationError("Risk Aversion is out of bounds (1,inf)")

    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        logger.info("CRRA OPTIMIZATION SUCCESSFUL")
        return result.x
    else:
        logger.error("CRRA OPTIMIZATION FAILED")
        raise OptimizationError("CRRA Optimization failed")

# Entropic Risk Measure Model
def ERM(w, THETA, data):
    logger.info("ERM OPTIMIZATION INITIATED")

    # ERM Objective 
    # Minimize 1/theta * log E[exp(-theta * X)]
    def f(w):
        X = data @ w
        return 1/THETA * np.log(np.mean(np.exp(-THETA * X)))

    # Robustness check
    if THETA <= 0:
        logger.error("INVALID ERM RISK AVERSION")
        raise OptimizationError("Risk aversion is out of bounds (0,inf)")

    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        logger.info("ERM OPTIMIZATION SUCCESSFUL")
        return result.x
    else:
        logger.error("ERM OPTIMIZATION FAILED")
        raise OptimizationError("ERM Optimization failed")

# Quadratic Utility Model
def quadraticutility(w, GAMMA, data):
    logger.info("QUADRATIC UTILITY OPTIMIZATION INITIATED")

    # Quadratic Utility Objective 
    # Minimize E[(gamma/2)W^2 - W]
    def f(w):
        X = 1 + np.maximum((data @ w), -1)
        return np.mean(GAMMA/2 * (X ** 2) - X)

    # Robustness check
    if GAMMA < 0:
        logger.error("INVALID RISK AVERSION")
        raise OptimizationError("Risk Aversion is out of bounds [0,inf)")

    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        logger.info("QUADRATIC UTILITY OPTIMIZATION SUCCESSFUL")
        return result.x
    else:
        logger.error("QUADRATIC UTILITY OPTIMIZATION FAILED")
        raise OptimizationError("Quadratic Utility Optimization failed")

# Entropic Value-at-Risk (EVaR) model
def EVaR(w, ALPHA, data):
    
    logger.info("EVAR OPTIMIZATION INITIATED")

    # EVaR Objective 
    # Minimize 1/s * {log E[exp(-sX)] - log (1 - ALPHA)}
    def f(x):

        # Extracting weights and auxiliary variable
        w = x[:-1]
        s = x[-1]

        X = data @ w
        return (1/s) * (np.log(np.mean(np.exp(-s * X))) - np.log(1 - ALPHA))

    # Robustness check
    if ALPHA > 1 or ALPHA < 0:
        logger.error("INVALID EVAR CONFIDENCE")
        raise OptimizationError("Confidence is out of bounds (0,1)")
    
    S = 1
    x0 = np.append(w, S)

    result = minimize(f, x0, method='SLSQP', bounds=[(0,1)]*len(w) + [(1e-8,None)], constraints= [{'type':'eq','fun': lambda x: x[:-1].sum()-1}])
    if result.success:
        logger.info("EVAR OPTIMIZATION SUCCESSFUL")
        return result.x[:-1]
    else:
        logger.error("EVAR OPTIMIZATION FAILED")
        raise OptimizationError("EVaR Optimization failed")

# The Best Portfolio
def Equal(w):
    return np.ones(len(w)) / len(w)