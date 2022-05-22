import numpy as np

from _base import BaseOpt


class RMSpropGraves(BaseOpt):
    """RMSpropGraves optimizer class.

    Attributes:
        eta (Callable): Learning rate.
                        If float comming, make lambda function
                        which return the value.
        rho (Callable): Oblivion rate.
                        If float comming, make lambda function
                        which return the value.

    Examples:
    >>> import numpy as np
    >>> obj = RMSpropGraves()
    >>> obj.update(grad=np.array([-0.5, 1])) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: <<Error sentence>>
    >>> obj.build()
    >>> obj.update(grad=np.array([-0.5, 1]))
    array([ 0.00045883, -0.00045883])
    >>> obj = RMSpropGraves(eta=lambda t=1: 1e-2 * 0.5**(t-1),
    ...                     rho=lambda t=1: 0.99 if t <= 5 else 0.5)
    >>> obj.build()
    >>> for i in range(1, 11):
    ...     print(f"t={i:2d}", obj.update(grad=np.array([-0.5, 1]), t=i))
    t= 1 [ 0.10050358 -0.10050373]
    t= 2 [ 0.03580204 -0.03580207]
    t= 3 [ 0.01472656 -0.01472657]
    t= 4 [ 0.00642494 -0.00642495]
    t= 5 [ 0.00289501 -0.00289501]
    t= 6 [ 0.00062575 -0.00062575]
    t= 7 [ 0.00036704 -0.00036704]
    t= 8 [ 0.00024139 -0.00024139]
    t= 9 [ 0.00016521 -0.00016521]
    t=10 [ 0.00011502 -0.00011502]
    """

    def __init__(self, *, eta=1e-4, rho=0.95, **kwds):
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

        self._m += (1-self.rho(**rhokwds))*(grad-self._m)
        self._v += (1-self.rho(**rhokwds))*(grad*grad - self._v)
        return -grad*self.eta(**etakwds)/np.sqrt(self._v - self._m*self._m)

    def build(self, *, _m=1e-8, _v=1e-8, **kwds):
        """Build optimizer."""
        self._m = _m
        self._v = _v
        self.update = self.__update
        self._is_built = True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
