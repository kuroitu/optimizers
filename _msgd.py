from enum import IntEnum
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
class MSGD(BaseOpt):
    """MSGD optimizer class.

    Examples:
    >>> import numpy as np
    >>> obj = MSGD()
    >>> print(obj)
    MSGD(eta=0.01, mu=0.9)
    >>> obj.update(np.array([-0.5, 1]))
    array([ 0.00050001, -0.00099999])
    """
    kind: InitVar[int] = 1
    eta: float = 1e-2
    mu: float = 0.9

    def update(self, grad, *args, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.

        Returns:
            delta (ndarray): Update delta.
        """
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
