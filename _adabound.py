from enum import IntEnum, auto
from typing import Callable, List
from dataclasses import dataclass, InitVar, field

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
class AdaBound(BaseOpt):
    """AdaBound optimizer class.

    Examples:
    >>> import numpy as np
    >>> obj = AdaBound()
    >>> print(obj)
    AdaBound(alpha=0.001, beta1=0.9, beta2=0.999, eta=0.1, eta_l_list=[0.1, 0.999], eta_u_list=[0.1, 0.999])
    >>> obj.update(np.array([-0.5, 1]))
    array([ 0.00316221, -0.00316226])
    """
    kind: InitVar[int] = 2
    alpha: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eta: float = 1e-1
    eta_l: Callable = field(
            default=lambda eta, beta2, t: eta*(1 - 1/((1-beta2)*t + 1)),
            repr=False)
    eta_u: Callable = field(
            default=lambda eta, beta2, t: eta*(1 + 1/((1-beta2)*t)),
            repr=False)
    eta_l_list: List = field(default_factory=list)
    eta_u_list: List = field(default_factory=list)

    def __post_init__(self, *args, **kwds):
        super().__post_init__(*args, **kwds)
        if not self.eta_l_list:
            self.eta_l_list = [self.eta, self.beta2]
        if not self.eta_u_list:
            self.eta_u_list = [self.eta, self.beta2]

    def update(self, grad, *args, t=1, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.
            t (int): Timestep.

        Returns:
            delta (ndarray): Update delta.
        """
        self._m += (1-self.beta1)*(grad-self._m)
        self._v += (1-self.beta2)*(grad*grad - self._v)
        eta_hat = np.clip(self.alpha/np.sqrt(self._v),
                          self.eta_l(*self.eta_l_list, t),
                          self.eta_u(*self.eta_u_list, t))
        delta = -eta_hat/np.sqrt(t)*self._m
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
