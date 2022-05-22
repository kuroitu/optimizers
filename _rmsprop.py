import numpy as np

from _base import BaseOpt


class RMSprop(BaseOpt):
    """RMSprop optimizer class.

    Attributes:
        eta (Callable): Learning rate.
                        If float comming, make lambda function
                        which return the value.
        rho (Callable): Oblivion rate.
                        If float comming, make lambda function
                        which return the value.

    Examples:
    >>> import numpy as np
    >>> obj = RMSprop()
    >>> obj.update(grad=np.array([-0.5, 1])) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: <<Error sentence>>
    >>> obj.build()
    >>> obj.update(grad=np.array([-0.5, 1]))
    array([ 0.0999998 , -0.09999995])
    >>> obj = RMSprop(eta=lambda t=1: 1e-2 * 0.5**(t-1),
    ...               rho=lambda t=1: 0.99 if t <= 5 else 0.5)
    >>> obj.build()
    >>> for i in range(1, 11):
    ...     print(f"t={i:2d}", obj.update(grad=np.array([-0.5, 1]), t=i))
    t= 1 [ 0.0999998  -0.09999995]
    t= 2 [ 0.03544403 -0.03544405]
    t= 3 [ 0.01450622 -0.01450622]
    t= 4 [ 0.00629709 -0.00629709]
    t= 5 [ 0.00282317 -0.00282318]
    t= 6 [ 0.00043149 -0.00043149]
    t= 7 [ 0.00017897 -0.00017897]
    t= 8 [ 8.32282879e-05 -8.32282881e-05]
    t= 9 [ 4.02778297e-05 -4.02778297e-05]
    t=10 [ 1.98281022e-05 -1.98281023e-05]
    """

    def __init__(self, *, eta=1e-2, rho=0.99, **kwds):
        super().__init__(**kwds)
        self._set_param2callable("eta", eta)
        self._set_param2callable("rho", rho)

    def __update(self, *, grad=None, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.

        Returns:
            delta (ndarray): Update delta.
        """
        etakwds = self._get_func_kwds(self.eta, kwds)
        rhokwds = self._get_func_kwds(self.rho, kwds)

        self._v += (1-self.rho(**rhokwds))*(grad*grad - self._v)
        return -grad*self.eta(**etakwds)/np.sqrt(self._v)

    def build(self, *, _v=1e-8, **kwds):
        """Build optimizer."""
        self._v = _v
        self.update = self.__update
        self._is_built = True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
