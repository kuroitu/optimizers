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
    u = auto()


@dataclass
class AdaMax(BaseOpt):
    """AdaMax optimizer class.

    Examples:
    >>> import numpy as np
    >>> obj = AdaMax()
    >>> print(obj)
    AdaMax(alpha=0.002, beta1=0.9, beta2=0.999)
    >>> obj.update(np.array([-0.5, 1]))
    array([ 0.002, -0.002])
    """
    kind: InitVar[int] = 2
    alpha: float = 2e-3
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
        self._u = np.maximum(self.beta2*self._u, np.abs(grad))
        alpha_t = self.alpha/(1 - self.beta1**t)
        delta = -alpha_t*self._m/self._u
        return delta

    @property
    def _m(self):
        return self.previous[_keys.m]

    @_m.setter
    def _m(self, value):
        self.previous[_keys.m] = value

    @property
    def _u(self):
        return self.previous[_keys.u]

    @_u.setter
    def _u(self, value):
        self.previous[_keys.u] = value


if __name__ == "__main__":
    import doctest
    doctest.testmod()
