import numpy as np
from opes.errors import PortfolioError

# Regularizer finding function
def find_regularizer(reg):
    regularizer = {
        None: lambda w: 0,
        "l1": lambda w: np.sum(np.abs(w)),
        "l2": lambda w: np.sum(w**2),
        "l-inf": lambda w: np.max(np.abs(w)),
        "entropy": lambda w: np.sum(np.abs(w) * np.log(np.abs(w) + 1e-12)),
        "variance": lambda w: np.var(w) if len(w) >= 2 else 0,
        "mpad": lambda w: np.mean(np.abs(w[:, None] - w[None, :])),
    }
    reg = str(reg).lower() if reg is not None else reg
    if reg in regularizer:
        return regularizer[reg]
    else:
        raise PortfolioError(f"Unknown regularizer: {reg}")