import numpy as np

from _base import BaseOpt


class SMORMS3(BaseOpt):
    """SMORMS3 optimizer class.

    Attributes:
        eta (Callable): Learning rate.
                        If float comming, make lambda function
                        which return the value.

    Examples:
    >>> import numpy as np
    >>> obj = SMORMS3()
    >>> obj.update(grad=np.array([-0.5, 1])) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: <<Error sentence>>
    >>> obj.build()
    >>> obj.update(grad=np.array([-0.5, 1]))
    array([ 0.00141421, -0.00141421])
    >>> obj = SMORMS3(eta=lambda t=1: 1e-2 * 0.5**(t-1))
    >>> obj.build()
    >>> for i in range(1, 11):
    ...     print(f"t={i:2d}", obj.update(grad=np.array([-0.5, 1]), t=i))
    t= 1 [ 0.01414214 -0.01414214]
    t= 2 [ 0.00547723 -0.00547723]
    t= 3 [ 0.00257248 -0.00257248]
    t= 4 [ 0.00126515 -0.00126515]
    t= 5 [ 0.00062862 -0.00062862]
    t= 6 [ 0.00031339 -0.00031339]
    t= 7 [ 0.00015647 -0.00015647]
    t= 8 [ 7.81799217e-05 -7.81799217e-05]
    t= 9 [ 3.90762038e-05 -3.90762038e-05]
    t=10 [ 1.95346726e-05 -1.95346726e-05]
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

        rho = 1/(1+self._s)
        self._s += 1 - self._zeta*self._s
        self._m += (1-rho)*(grad - self._m)
        self._v += (1-rho)*(grad*grad - self._v)
        self._zeta = (self._m*self._m/self._v)
        return -grad*np.minimum(self.eta(**etakwds), self._zeta) \
                    /np.sqrt(self._v)

    def build(self, *, _s=1, _m=1e-8, _v=1e-8, _zeta=1e-8, **kwds):
        """Build optimizer."""
        self._s = _s
        self._m = _m
        self._v = _v
        self._zeta = _zeta
        self.update = self.__update
        self._is_built = True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
