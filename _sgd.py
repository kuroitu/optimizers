import numpy as np

from _base import BaseOpt


class SGD(BaseOpt):
    """SGD optimizer class.

    Attributes:
        eta (Callable): Learning rate.
                        If float comming, make lambda function
                        which return the value.

    Examples:
    >>> import numpy as np
    >>> obj = SGD(eta=1e-2)
    >>> obj.update(grad=np.array([-0.5, 1])) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: <<Error sentence>>
    >>> obj.build()
    >>> obj.update(grad=np.array([-0.5, 1]))
    array([ 0.005, -0.01 ])
    >>> obj = SGD(eta=lambda t=1: 1e-2 * 0.5**(t-1))
    >>> obj.build()
    >>> for i in range(1, 11):
    ...     print(f"t={i:2d}", obj.update(grad=np.array([-0.5, 1]), t=i))
    t= 1 [ 0.005 -0.01 ]
    t= 2 [ 0.0025 -0.005 ]
    t= 3 [ 0.00125 -0.0025 ]
    t= 4 [ 0.000625 -0.00125 ]
    t= 5 [ 0.0003125 -0.000625 ]
    t= 6 [ 0.00015625 -0.0003125 ]
    t= 7 [ 7.8125e-05 -1.5625e-04]
    t= 8 [ 3.90625e-05 -7.81250e-05]
    t= 9 [ 1.953125e-05 -3.906250e-05]
    t=10 [ 9.765625e-06 -1.953125e-05]
    """

    def __init__(self, *, eta=1e-2, **kwds):
        super().__init__(**kwds)
        self._set_param2callable("eta", eta)

    def __update(self, *, grad=None, **kwds):
        """Compute update delta.

        Args:
            grad (ndarray): Gradient propagating from the next layer.

        Returns:
            delta (ndarray): Update delta.
        """
        etakwds = self._get_func_kwds(self.eta, kwds)

        return -self.eta(**etakwds)*grad

    def build(self, *args, **kwds):
        """Build optimizer."""
        self.update = self.__update
        self._is_built = True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
