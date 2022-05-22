from enum import IntEnum, auto
from typing import Callable
from dataclasses import dataclass, InitVar, field

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseOpt
except ImportError:
    # For doctest
    from main.dl.opt import BaseOpt


class _keys(IntEnum):
    alpha = 0
    u = auto()
    v = auto()
    g = auto()


@dataclass
class SantaE(BaseOpt):
    """SantaE optimizer class.

    Examples:
    >>> import numpy as np
    >>> obj = SantaE()
    >>> print(obj)
    SantaE(eta=0.01, sigma=0.95, anne_rate=0.5, burnin=100, C=5, N=16)
    """
    kind: InitVar[int] = 4
    eta: float = 1e-2
    sigma:float = 0.95
    #lambda_: float = 1e-8
    anne_func: Callable = field(default=lambda t, rate: t**rate, repr=False)
    anne_rate: float = 0.5
    burnin: int = 100
    C: int = 5
    N: int = 16

    def __post_init__(self, *args, **kwds):
        super().__post_init__(*args, **kwds)
        self.previous[_keys.alpha] = np.sqrt(self.eta)*self.C
        self.previous[_keys.u] \
                = (np.sqrt(self.eta)
                   *np.random.randn(*self.previous[_keys.u].shape))

    def update(self, grad, *args, t=1, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.
            t (int): Timestep.

        Returns:
            delta (ndarray): Update delta.
        """
        self._v += (1-self.sigma)*(grad * grad / self.N**2 - self._v)
        #g_t = 1/np.sqrt(self.lambda_+np.sqrt(self._v))
        g_t = 1/np.sqrt(np.sqrt(self._v))
        eta_div_beta = self.eta/self.anne_func(t, self.anne_rate)
        if t < self.burnin:
            self._alpha += self._u*self._u - eta_div_beta
            u_t = eta_div_beta*(1 - self._g/g_t)/self._u
            u_t += (np.sqrt(2*eta_div_beta*self._g)
                    *np.random.randn(*self._u.shape))
        else:
            u_t = 0
        self._g = g_t
        self._u += u_t - self._alpha*self._u - self.eta*self._g*grad
        delta = self._g*self._u
        return delta

    @property
    def _alpha(self):
        return self.previous[_keys.alpha]

    @_alpha.setter
    def _alpha(self, value):
        self.previous[_keys.alpha] = value

    @property
    def _u(self):
        return self.previous[_keys.u]

    @_u.setter
    def _u(self, value):
        self.previous[_keys.u] = value

    @property
    def _v(self):
        return self.previous[_keys.v]

    @_v.setter
    def _v(self, value):
        self.previous[_keys.v] = value

    @property
    def _g(self):
        return self.previous[_keys.g]

    @_g.setter
    def _g(self, value):
        self.previous[_keys.g] = value


if __name__ == "__main__":
    import doctest
    doctest.testmod()
