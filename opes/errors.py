"""
Module for errors

Parent error: OpesError
Child errors: 
    - PortfolioError: When a portfolio variable is invalid or absent.
    - DataError: For data related problems.
    - OptimizationError: Denotes optimizer related issues (eg. Failed to converge).
"""

class OpesError(Exception):
    pass


class PortfolioError(OpesError):
    pass


class DataError(OpesError):
    pass


class OptimizationError(OpesError):
    pass
