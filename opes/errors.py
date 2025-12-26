class OpesError(Exception):
    """Base exception class for all OPES-related errors."""
    pass

class PortfolioError(OpesError):
    """Raised when there is an error in portfolio configuration."""
    pass

class DataError(OpesError):
    """Raised when input data is invalid, insufficient, or inconsistent."""
    pass

class OptimizationError(OpesError):
    """Raised when the portfolio optimization process fails."""
    pass