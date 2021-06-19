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
    s = auto()


@dataclass
class AdaBelief(BaseOpt):
    """AdaBelief optimizer class.

    Examples:
    >>> import numpy as np
    >>> obj = AdaBelief()
    >>> print(obj)
    AdaBelief(alpha=0.001, beta1=0.9, beta2=0.999)
    >>> obj.update(np.array([-0.5, 1]))
    array([ 0.00099998, -0.001     ])
    """
    kind: InitVar[int] = 2
    alpha: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999

    def update(self, grad, *args, t=1, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.
            t (int): Timestep.

        Returns:
            delta (ndarray): Update delta.
        """
        self._m += (1-self.beta1)*(grad-self._m)
        self._s += (1-self.beta2)*((grad-self._m)**2 - self._s)
        alpha_t = self.alpha*np.sqrt(1 - self.beta2**t)/(1 - self.beta1**t)
        delta = -alpha_t*self._m/np.sqrt(self._s)
        return delta

    @property
    def _m(self):
        return self.previous[_keys.m]

    @_m.setter
    def _m(self, value):
        self.previous[_keys.m] = value

    @property
    def _s(self):
        return self.previous[_keys.s]

    @_s.setter
    def _s(self, value):
        self.previous[_keys.s] = value


if __name__ == "__main__":
    import doctest
    doctest.testmod()
