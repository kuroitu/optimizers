import sys
from enum import IntEnum
from typing import Any, Union
from dataclasses import dataclass, InitVar

import numpy as np

try:
    from ._base import BaseOpt
except ImportError:
    # For doctest
    from main.dl.opt import BaseOpt


class _keys(IntEnum):
    delta = 0


@dataclass
class NAG(BaseOpt):
    """NAG optimizer class.

    Examples:
    #>>> import numpy as np
    #>>> obj = NAG()
    #>>> print(obj)
    #NAG(delta=array([1.e-08, 1.e-08]), eta=0.01, mu=0.9)
    #>>> obj.update(np.array([-0.5, 1]))
    #array([ 0.00050001, -0.00099999])
    """
    kind: InitVar[int] = 1
    parent: Any = None
    eta: float = 1e-2
    mu: float = 0.9

    def update(self, grad, x, *args, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.
            x (ndarray): Input array.

        Returns:
            delta (ndarray): Update delta.
        """
        # Repropagation.
        self.parent.params += self.mu*self._delta
        _ = self.parent.forward(x, *args, **kwds)
        grad = self.parent.backward(grad, *args, **kwds)
        self.parent.params -= self.mu*self._delta
        # Compute update delta.
        delta = self._delta = self.mu*self._delta - (1-self.mu)*self.eta*grad
        return delta

    @property
    def _delta(self):
        return self.previous[_keys.delta]

    @_delta.setter
    def _delta(self, value):
        self.previous[_keys.delta] = value


if __name__ == "__main__":
    import doctest
    doctest.testmod()
