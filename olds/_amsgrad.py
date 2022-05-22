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
    v_hat = auto()


@dataclass
class AMSGrad(BaseOpt):
    """AMSGrad optimizer class.

    Examples:
    >>> import numpy as np
    >>> obj = AMSGrad()
    >>> print(obj)
    AMSGrad(alpha=0.001, beta1=0.9, beta2=0.999)
    >>> obj.update(np.array([-0.5, 1]))
    array([ 0.00316221, -0.00316226])
    """
    kind: InitVar[int] = 3
    alpha: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999

    def update(self, grad, *args, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.

        Returns:
            delta (ndarray): Update delta.
        """
        self._m += (1-self.beta1)*(grad-self._m)
        self._v += (1-self.beta2)*(grad*grad - self._v)
        self._v_hat = np.maximum(self._v_hat, self._v)
        delta = -self.alpha*self._m/np.sqrt(self._v_hat)
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
    def _v_hat(self):
        return self.previous[_keys.v_hat]

    @_v_hat.setter
    def _v_hat(self, value):
        self.previous[_keys.v_hat] = value


if __name__ == "__main__":
    import doctest
    doctest.testmod()
