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
class Nadam(BaseOpt):
    """Nadam optimizer class.

    Examples:
    >>> import numpy as np
    >>> obj = Nadam()
    >>> print(obj)
    Nadam(alpha=0.002, mu=0.975, nu=0.999)
    >>> obj.update(np.array([-0.5, 1]))
    array([ 0.00298878, -0.00298882])
    """
    kind: InitVar[int] = 2
    alpha: float = 2e-3
    mu: float = 0.975
    nu: float = 0.999

    def update(self, grad, *args, t=1, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.
            t (int): Timestep.

        Returns:
            delta (ndarray): Update delta.
        """
        self._m += (1-self.mu)*(grad - self._m)
        self._v += (1-self.nu)*(grad*grad - self._v)
        m_hat = (self._m*self.mu/(1 - self.mu**(t+1))
                 + grad*(1-self.mu)/(1 - self.mu**t))
        v_hat = self._v*self.nu/(1 - self.nu**t)
        delta = -self.alpha*m_hat/np.sqrt(v_hat)
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
