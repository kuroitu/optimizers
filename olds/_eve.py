from enum import IntEnum, auto
from typing import Union
from dataclasses import dataclass, InitVar

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseOpt
except ImportError:
    # For doctest
    from main.dl.opt import BaseOpt


class _keys(IntEnum):
    m = 0
    v = auto()
    f = auto()
    d_tilde = auto()


@dataclass
class Eve(BaseOpt):
    """Eve optimizer class.

    Examples:
    >>> import numpy as np
    >>> obj = Eve()
    >>> print(obj)
    Eve(alpha=0.001, beta1=0.9, beta2=0.999, beta3=0.999, c=10, f_star=0)
    >>> obj.update(np.array([-0.5, 1]))
    array([ 0.00099998, -0.001     ])
    """
    kind: InitVar[int] = 4
    alpha: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    beta3: float = 0.999
    c: float = 10
    f_star: Union[float, ndarray] = 0

    def update(self, grad, *args, t=1, f=1, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.
            t (int): Timestep.
            f (float): Current objective value.

        Returns:
            delta (ndarray): Update delta.
        """
        self._m += (1-self.beta1)*(grad-self._m)
        self._v += (1-self.beta2)*(grad*grad - self._v)
        m_hat = self._m/(1 - self.beta1**t)
        v_hat = self._v/(1 - self.beta2**t)
        if t > 1:
            d = (np.abs(self._f-self.f_star)
                 /(np.minimum(self._f, f)-self.f_star))
            d_hat = np.clip(d, 1/self.c, self.c)
            self._d_tilde += (1-self.beta3)*(d_hat-self._d_tilde)
        else:
            self._d_tilde = 1
            self._f = f
        delta = -self.alpha/self._d_tilde*m_hat/np.sqrt(v_hat)
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

    @property
    def _f(self):
        return self.previous[_keys.f]

    @_f.setter
    def _f(self, value):
        self.previous[_keys.f] = value

    @property
    def _d_tilde(self):
        return self.previous[_keys.d_tilde]

    @_d_tilde.setter
    def _d_tilde(self, value):
        self.previous[_keys.d_tilde] = value


if __name__ == "__main__":
    import doctest
    doctest.testmod()
