import sys
from typing import Any, Union, Dict
from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray

from main.dl.layer import BaseLayer
try:
    from ._base import BaseOpt
except ImportError:
    # For doctest
    from main.dl.opt import BaseOpt


@dataclass
class SAM(BaseOpt):
    """SAM optimizer class."""
    #parent: Any = None
    parent: BaseLayer = None
    q: int = 2
    rho: float = 5e-2
    opt: Union[str, BaseOpt] = "adam"
    opt_dict: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self, *args, **kwds):
        from ._interface import get_opt

        super().__post_init__(*args, **kwds)
        self.opt = get_opt(self.opt, **self.opt_dict)

    def update(self, grad, x, *args, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.
            x (ndarray): Input array.

        Returns:
            delta (ndarray): Update delta.
        """
        eps = self._epsilon(grad)
        self.parent.params += eps
        _ = self.parent.forward(x, *args, **kwds)
        grad = self.parent.backward(grad, *args, **kwds)
        self.parent.params -= eps
        return self.opt.update(grad, *args, **kwds)

    def _epsilon(self, grad):
        return (self.rho * np.sign(grad) * np.abs(grad)**(self.q-1)
                / np.linalg.norm(
                    grad if isinstance(grad, ndarray) else np.array([grad]),
                    ord=self.q)**(1 - 1/self.q))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
