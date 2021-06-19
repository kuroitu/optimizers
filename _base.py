from dataclasses import dataclass, InitVar, field

import numpy as np
from numpy import ndarray


@dataclass
class BaseOpt():
    """Base class for optimizers.

    Args:
        n (int): Number of parameters to update.
        eps (float): Tiny values to avoid division by zero.
    """
    kind: InitVar[int] = 0
    n: InitVar[int] = 2
    eps: InitVar[float] = 1e-8
    previous: ndarray = field(init=False, repr=False)

    def __post_init__(self, kind, n, eps, *args, **kwds):
        if kind:
            self.previous = np.full((kind, n), eps)
        else:
            self.previous = None

    def update(self, *args, **kwds):
        raise NotImplementedError("'update' method must be implemented.")
