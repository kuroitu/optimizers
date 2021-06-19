from enum import IntEnum, auto
from dataclasses import dataclass, InitVar

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseOpt
except ImportError:
    # For doctest
    from main.dl.opt import BaseOpt


class _keys(IntEnum):
    s = 0
    m = auto()
    v = auto()
    zeta = auto()


@dataclass
class SMORMS3(BaseOpt):
    """SMORMS3 optimizer class.

    Examples:
    >>> import numpy as np
    >>> obj = SMORMS3()
    >>> print(obj)
    SMORMS3(eta=0.001)
    >>> obj.update(np.array([-0.5, 1]))
    array([ 0.00141421, -0.00141421])
    """
    kind: InitVar[int] = 4
    eta: float = 1e-3

    def __post_init__(self, *args, **kwds):
        super().__post_init__(*args, **kwds)
        self.previous[_keys.s] = 1

    def update(self, grad, *args, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.

        Returns:
            delta (ndarray): Update delta.
        """
        rho = 1/(1+self._s)
        self._s += 1 - self._zeta*self._s
        self._m += (1-rho)*(grad - self._m)
        self._v += (1-rho)*(grad*grad - self._v)
        self._zeta = (self._m*self._m/self._v)
        delta = -grad*np.minimum(self.eta, self._zeta)/np.sqrt(self._v)
        return delta

    @property
    def _s(self):
        return self.previous[_keys.s]

    @_s.setter
    def _s(self, value):
        self.previous[_keys.s] = value

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

    @property
    def _zeta(self):
        return self.previous[_keys.zeta]

    @_zeta.setter
    def _zeta(self, value):
        self.previous[_keys.zeta] = value


if __name__ == "__main__":
    import doctest
    doctest.testmod()
