from dataclasses import dataclass

import numpy as np

try:
    from ._base import BaseOpt
except ImportError:
    # For doctest
    from main.dl.opt import BaseOpt


@dataclass
class SGD(BaseOpt):
    """SGD optimizer class.

    Examples:
    >>> import numpy as np
    >>> obj = SGD()
    >>> print(obj)
    SGD(eta=0.01)
    >>> obj.update(np.array([-0.5, 1]))
    array([ 0.005, -0.01 ])
    """
    eta: float = 1e-2

    def update(self, grad, *args, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.

        Returns:
            delta (ndarray): Update delta.
        """
        delta = -self.eta*grad
        return delta


if __name__ == "__main__":
    import doctest
    doctest.testmod()
