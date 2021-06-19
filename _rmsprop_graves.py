from enum import IntEnum, auto
from dataclasses import dataclass, InitVar

import numpy as np

try:
    from ._base import BaseOpt
except ImportError:
    # For doctest
    from main.dl.opt import BaseOpt


class _keys(IntEnum):
    m = 0
    v = auto()


@dataclass
class RMSpropGraves(BaseOpt):
    """RMSpropGraves optimizer class.

    Examples:
    >>> import numpy as np
    >>> obj = RMSpropGraves()
    >>> print(obj)
    RMSpropGraves(eta=0.0001, rho=0.95)
    >>> obj.update(np.array([-0.5, 1]))
    array([ 0.00045692, -0.00045842])
    """
    kind: InitVar[int] = 2
    eta: float = 1e-4
    rho: float = 0.95

    def update(self, grad, *args, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.

        Returns:
            delta (ndarray): Update delta.
        """
        self._m += (1-self.rho)*(grad-self._m)
        self._v += (1-self.rho)*(grad*grad - self._v)
        delta = -grad*self.eta/np.sqrt(self._v - self._m*self._m)
        return delta

    @property
    def _m(self):
        return self.previous[_keys.m]

    @_m.setter
    def _m(self, value):
        self.previous[_keys.m] = value

    @property
    def _v(self):
        return self.previous[_keys.v]

    @_v.setter
    def _v(self, value):
        self.previous[_keys.v] = value


if __name__ == "__main__":
    import doctest
    doctest.testmod()
