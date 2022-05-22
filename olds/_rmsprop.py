from enum import IntEnum
from dataclasses import dataclass, InitVar

import numpy as np

try:
    from ._base import BaseOpt
except ImportError:
    # For doctest
    from main.dl.opt import BaseOpt


class _keys(IntEnum):
    v = 0


@dataclass
class RMSprop(BaseOpt):
    """RMSprop optimizer class.

    Examples:
    >>> import numpy as np
    >>> obj = RMSprop()
    >>> print(obj)
    RMSprop(eta=0.01, rho=0.99)
    >>> obj.update(np.array([-0.5, 1]))
    array([ 0.0999998 , -0.09999995])
    """
    kind: InitVar[int] = 1
    eta: float = 1e-2
    rho: float = 0.99

    def update(self, grad, *args, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.

        Returns:
            delta (ndarray): Update delta.
        """
        self._v += (1-self.rho)*(grad*grad - self._v)
        delta = -grad*self.eta/np.sqrt(self._v)
        return delta

    @property
    def _v(self):
        return self.previous[_keys.v]

    @_v.setter
    def _v(self, value):
        self.previous[_keys.v] = value


if __name__ == "__main__":
    import doctest
    doctest.testmod()
