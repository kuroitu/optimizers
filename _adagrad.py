import numpy as np

from _base import BaseOpt


class AdaGrad(BaseOpt):
    """AdaGrad optimizer class.

    Attributes:
        eta (Callable): Learning rate.
                        If float comming, make lambda function
                        which return the value.

    Examples:
    >>> import numpy as np
    >>> obj = AdaGrad()
    >>> obj.update(grad=np.array([-0.5, 1])) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: <<Error sentence>>
    >>> obj.build()
    >>> obj.update(grad=np.array([-0.5, 1]))
    array([ 0.001, -0.001])
    >>> obj = AdaGrad(eta=lambda t=1: 1e-3 * 0.5**(t-1))
    >>> obj.build()
    >>> for i in range(1, 11):
    ...     print(f"t={i:2d}", obj.update(grad=np.array([-0.5, 1]), t=i))
    t= 1 [ 0.001 -0.001]
    t= 2 [ 0.00035355 -0.00035355]
    t= 3 [ 0.00014434 -0.00014434]
    t= 4 [ 6.24999997e-05 -6.24999999e-05]
    t= 5 [ 2.79508496e-05 -2.79508497e-05]
    t= 6 [ 1.27577590e-05 -1.27577591e-05]
    t= 7 [ 5.90569487e-06 -5.90569489e-06]
    t= 8 [ 2.76213586e-06 -2.76213586e-06]
    t= 9 [ 1.30208333e-06 -1.30208333e-06]
    t=10 [ 6.17632354e-07 -6.17632355e-07]
    """

    def __init__(self, *, eta=1e-3, **kwds):
        super().__init__(**kwds)
        self._set_param2callable("eta", eta)

    def __update(self, *, grad=None, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.

        Returns:
            delta (ndarray): Update delta.
        """
        etakwds = self._get_func_kwds(self.eta, kwds)

        self._g += grad*grad
        return -grad*self.eta(**etakwds)/np.sqrt(self._g)

    def build(self, *args, _g=1e-8, **kwds):
        """Build optimizer."""
        self._g = _g
        self.update = self.__update
        self._is_built = True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
