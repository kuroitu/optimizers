from enum import IntEnum
from dataclasses import dataclass, InitVar

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseOpt
except ImportError:
    # For doctest
    from main.dl.opt import BaseOpt


class _keys(IntEnum):
    sum_g = 0


@dataclass
class AdaGrad(BaseOpt):
    """AdaGrad optimizer class.

    Examples:
    >>> import numpy as np
    >>> obj = AdaGrad()
    >>> print(obj)
    AdaGrad(eta=0.001)
    >>> obj.update(np.array([-0.5, 1]))
    array([ 0.001, -0.001])
    """
    kind: InitVar[int] = 1
    eta: float = 1e-3

    def update(self, grad, *args, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.

        Returns:
            delta (ndarray): Update delta.
        """
        self._sum_g += grad*grad
        delta = -grad*self.eta/np.sqrt(self._sum_g)
        return delta

    @property
    def _sum_g(self):
        return self.previous[_keys.sum_g]

    @_sum_g.setter
    def _sum_g(self, value):
        self.previous[_keys.sum_g] = value


if __name__ == "__main__":
    import doctest
    doctest.testmod()
