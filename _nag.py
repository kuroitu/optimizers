import numpy as np

from _base import BaseOpt


class NAG(BaseOpt):
    """NAG optimizer class.

    Attributes:
        eta (Callable): Learning rate.
                        If float comming, make lambda function
                        which return the value.
        mu (Callable): Oblivion rate.
                       If float comming, make lambda function
                       which return the value.

    Examples:
    >>> import numpy as np
    >>> class NAGTest():
    ...     def __init__(self, *args, x=None, **kwds):
    ...         self.params = x
    ...
    ...     def forward(self, *, x=None, **kwds):
    ...         if x is not None:
    ...             self.params = x
    ...         return self.params**2
    ...
    ...     def backward(self, *, grad=None, **kwds):
    ...         return 2*self.params*grad
    >>> parent = NAGTest(x=np.array([-0.5, 1]))
    >>> obj = NAG()
    >>> obj.update(grad=np.array([-0.5, 1]),
    ...            parent=parent) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: <<Error sentence>>
    >>> obj.build()
    >>> obj.update(grad=np.array([1, -2]), parent=parent)
    array([0.00100001, 0.00400001])
    >>> obj = NAG(eta=lambda t=1: 1e-2 * 0.5**(t-1),
    ...           mu=lambda t=1: 0.9 if t <= 5 else 0.5)
    >>> obj.build()
    >>> for i in range(1, 11):
    ...     print(f"t={i:2d}", obj.update(grad=np.array([-0.5, 1]),
    ...                                   parent=parent, t=i))
    t= 1 [-0.00049999 -0.00199999]
    t= 2 [-0.00070022 -0.00279819]
    t= 3 [-0.00075535 -0.00301711]
    t= 4 [-0.0007424  -0.00296472]
    t= 5 [-0.00069945 -0.00279292]
    t= 6 [-0.00042791 -0.00170852]
    t= 7 [-0.00025303 -0.00101038]
    t= 8 [-0.00014605 -0.00058327]
    t= 9 [-8.27932795e-05 -3.30688297e-04]
    t=10 [-4.62798565e-05 -1.84872169e-04]
    """

    def __init__(self, *, eta=1e-2, mu=0.9, **kwds):
        super().__init__(**kwds)
        self._set_param2callable("eta", eta)
        self._set_param2callable("mu", mu)

    def __update(self, *, grad=None, parent=None, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.
            parent (BaseLayer): Functions that this optimizer
                                tries to optimize.

        Returns:
            delta (ndarray): Update delta.
        """
        etakwds = self._get_func_kwds(self.eta, kwds)
        mukwds = self._get_func_kwds(self.mu, kwds)
        eta_t = self.eta(**etakwds)
        mu_t = self.mu(**mukwds)

        # Repropagation.
        parent.params += mu_t*self._delta
        _ = parent.forward(**kwds)
        grad = parent.backward(grad=grad, **kwds)
        parent.params -= mu_t*self._delta

        self._delta -= (1-mu_t)*(self._delta + eta_t*grad)
        return self._delta

    def build(self, *, _delta=1e-8, **kwds):
        """Build optimizer."""
        self._delta = _delta
        self.update = self.__update
        self._is_built = True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
