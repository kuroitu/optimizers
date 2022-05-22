import numpy as np

from _base import BaseOpt


class AdaDelta(BaseOpt):
    """AdaDelta optimizer class.

    Attributes:
        rho (Callable): Oblivion rate.
                        If float comming, make lambda function
                        which return the value.

    Examples:
    >>> import numpy as np
    >>> obj = AdaDelta()
    >>> obj.update(grad=np.array([-0.5, 1])) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: <<Error sentence>>
    >>> obj.build()
    >>> obj.update(grad=np.array([-0.5, 1]))
    array([ 0.00044721, -0.00044721])
    >>> obj = AdaDelta(rho=lambda t=1: 0.95 if t <= 5 else 0.5)
    >>> obj.build()
    >>> for i in range(1, 11):
    ...     print(f"t={i:2d}", obj.update(grad=np.array([-0.5, 1]), t=i))
    t= 1 [ 0.00044721 -0.00044721]
    t= 2 [ 0.00044721 -0.00044721]
    t= 3 [ 0.00044721 -0.00044721]
    t= 4 [ 0.00044721 -0.00044721]
    t= 5 [ 0.00044721 -0.00044721]
    t= 6 [ 0.00029396 -0.00029396]
    t= 7 [ 0.00029396 -0.00029396]
    t= 8 [ 0.00029396 -0.00029396]
    t= 9 [ 0.00029396 -0.00029396]
    t=10 [ 0.00029396 -0.00029396]
    """

    def __init__(self, *, rho=0.95, **kwds):
        super().__init__(**kwds)
        self._set_param2callable("rho", rho)

    def __update(self, *, grad=None, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.

        Returns:
            delta (ndarray): Update delta.
        """
        rhokwds = self._get_func_kwds(self.rho, kwds)
        rho_t = self.rho(**rhokwds)

        self._v += (1-rho_t)*(grad*grad - self._v)
        delta = -grad*np.sqrt(self._u)/np.sqrt(self._v)
        self._u += (1-rho_t)*(delta*delta - self._u)
        return delta

    def build(self, *, _v=1e-8, _u = 1e-8, **kwds):
        """Build optimizer."""
        self._v = _v
        self._u = _u
        self.update = self.__update
        self._is_built = True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
