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
    delta = auto()


@dataclass
class SantaSSS(BaseOpt):
    """SantaSSS optimizer class.

    Examples:
    >>> import numpy as np
    >>> obj = SantaSSS()
    >>> print(obj)
    SantaSSS(eta=0.01, sigma=0.95, anne_rate=0.5, burnin=100, C=5, N=16)
    """
    kind: InitVar[int] = 5
    eta: float = 1e-1
    sigma:float = 0.95
    #lambda_: float = 1e-8
    anne_func: Callable = field(default=lambda t, rate: t**rate, repr=False)
    anne_rate: float = 0.5
    burnin: int = 100
    C: int = 5
    N: int = 16

    def __post_init__(self, *args, **kwds):
        super().__post_init__(*args, **kwds)
        self._alpha = np.sqrt(self.eta)*self.C
        self.previous[[_keys.u, _keys.delta]] \
                = (np.sqrt(self.eta)
                   *np.random.randn(
                       *self.previous[[_keys.u, _keys.delta]].shape))

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
        self._delta = 0.5*g_t*self._u
        eta_div_beta = self.eta/self.anne_func(t, self.anne_rate)
        if t < self.burnin:
            self._alpha += 0.5*(self._u*self._u - eta_div_beta)
            u_t = np.exp(-0.5*self._alpha)*self._u
            u_t -= self.eta*g_t*grad
            u_t += (np.sqrt(2*eta_div_beta*self._g)
                    * np.random.randn(*self._u.shape))
            u_t += eta_div_beta*(1 - self._g/g_t)/self._u
            self._u = np.exp(-0.5*self._alpha)*u_t
            self._alpha += 0.5*(self._u*self._u - eta_div_beta)
        else:
            self._u *= np.exp(-0.5*self._alpha)
            self._u -= self.eta*g_t*grad
            self._u *= np.exp(-0.5*self._alpha)
        self._g = g_t
        self._delta += 0.5*self._g*self._u
        delta = self._delta
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

    @property
    def _delta(self):
        return self.previous[_keys.delta]

    @_delta.setter
    def _delta(self, value):
        self.previous[_keys.delta] = value


if __name__ == "__main__":
    import doctest
    doctest.testmod()
