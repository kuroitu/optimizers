import numpy as np

from _base import BaseOpt


class MSGD(BaseOpt):
    """MSGD optimizer class.

    Attributes:
        eta (Callable): Learning rate.
                        If float comming, make lambda function
                        which return the value.
        mu (Callable): Oblivion rate.
                       If float comming, make lambda function
                       which return the value.

    Examples:
    >>> import numpy as np
    >>> obj = MSGD(eta=1e-2, mu=0.9)
    >>> obj.update(grad=np.array([-0.5, 1])) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: <<Error sentence>>
    >>> obj.build()
    >>> obj.update(grad=np.array([-0.5, 1]))
    array([ 0.00050001, -0.00099999])
    >>> obj = MSGD(eta=lambda t=1: 1e-2 * 0.5**(t-1),
    ...            mu=lambda t=1: 0.9 if t <= 5 else 0.5)
    >>> obj.build()
    >>> for i in range(1, 11):
    ...     print(f"t={i:2d}", obj.update(grad=np.array([-0.5, 1]), t=i))
    t= 1 [ 0.00050001 -0.00099999]
    t= 2 [ 0.00070001 -0.00139999]
    t= 3 [ 0.00075501 -0.00150999]
    t= 4 [ 0.00074201 -0.00148399]
    t= 5 [ 0.00069906 -0.00139809]
    t= 6 [ 0.00042765 -0.0008553 ]
    t= 7 [ 0.00025289 -0.00050577]
    t= 8 [ 0.00014598 -0.00029195]
    t= 9 [ 8.27534941e-05 -1.65505881e-04]
    t=10 [ 4.62595595e-05 -9.25185655e-05]
    """

    def __init__(self, *, eta=1e-2, mu=0.9, **kwds):
        super().__init__(**kwds)
        self._set_param2callable("eta", eta)
        self._set_param2callable("mu", mu)

    def __update(self, *, grad=None, **kwds):
        """Compute update delta.

        Args:
            grad (ndarray): Gradient propagating from the next layer.

        Returns:
            delta (ndarray): Update delta.
                             Notes; must not change this
                                    because this is view of ndarray.
        """
        etakwds = self._get_func_kwds(self.eta, kwds)
        mukwds = self._get_func_kwds(self.mu, kwds)

        self._delta -= (1-self.mu(**mukwds)) \
                     * (self._delta + self.eta(**etakwds)*grad)
        return self._delta

    def build(self, *, _delta=1e-8, **kwds):
        """Build optimizer."""
        self._delta = _delta
        self.update = self.__update
        self._is_built = True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
