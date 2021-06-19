from enum import IntEnum, auto
from dataclasses import dataclass, InitVar

import numpy as np

try:
    from ._base import BaseOpt
except ImportError:
    # For doctest
    from main.dl.opt import BaseOpt


class _keys(IntEnum):
    v = 0
    u = auto()


@dataclass
class AdaDelta(BaseOpt):
    """AdaDelta optimizer class.

    Examples:
    >>> import numpy as np
    >>> obj = AdaDelta()
    >>> print(obj)
    AdaDelta(rho=0.95)
    >>> obj.update(np.array([-0.5, 1]))
    array([ 0.00447197, -0.00447209])
    """
    kind: InitVar[int] = 2
    rho: float = 0.95

    def update(self, grad, *args, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.

        Returns:
            delta (ndarray): Update delta.
        """
        self._v += (1-self.rho)*(grad*grad - self._v)
        delta = -grad*np.sqrt(self._u)/np.sqrt(self._v)
        self._u += (1-self.rho)*(delta*delta - self._u)
        return delta

    @property
    def _v(self):
        return self.previous[_keys.v]

    @_v.setter
    def _v(self, value):
        self.previous[_keys.v] = value

    @property
    def _u(self):
        return self.previous[_keys.u]

    @_u.setter
    def _u(self, value):
        self.previous[_keys.u] = value


if __name__ == "__main__":
    import doctest
    doctest.testmod()
