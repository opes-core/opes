class OpesError(Exception):
    pass

class PortfolioError(OpesError):
    pass

class DataError(OpesError):
    pass

class OptimizationError(OpesError):
    pass