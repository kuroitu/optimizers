import numpy as np

from _base import BaseOpt


class AdaMax(BaseOpt):
    """AdaMax optimizer class.

    Attributes:
        alpha (Callable): Learning rate.
                          If float comming, make lambda function
                          which return the value.
        beta1 (Callable): Momentum oblivion rate.
                          If float comming, make lambda function
                          which return the value.
        beta2 (Callable): Gradient oblivion rate.
                          If float comming, make lambda function
                          which return the value.

    Examples:
    >>> import numpy as np
    >>> obj = AdaMax()
    >>> obj.update(grad=np.array([-0.5, 1])) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: <<Error sentence>>
    >>> obj.build()
    >>> obj.update(grad=np.array([-0.5, 1]))
    array([ 0.002, -0.002])
    >>> obj = AdaMax(alpha=lambda t=1: 2e-3 * 0.5**(t-1),
    ...              beta1=lambda t=1: 0.9 if t <= 5 else 0.5,
    ...              beta2=lambda t=1: 0.999 if t <= 5 else 0.555)
    >>> obj.build()
    >>> for i in range(1, 11):
    ...     print(f"t={i:2d}", obj.update(grad=np.array([-0.5, 1]), t=i))
    t= 1 [ 0.002 -0.002]
    t= 2 [ 0.001 -0.001]
    t= 3 [ 0.0005 -0.0005]
    t= 4 [ 0.00025 -0.00025]
    t= 5 [ 0.000125 -0.000125]
    t= 6 [ 6.24999995e-05 -6.25000003e-05]
    t= 7 [ 3.12499999e-05 -3.12500001e-05]
    t= 8 [ 1.5625e-05 -1.5625e-05]
    t= 9 [ 7.81249999e-06 -7.81250000e-06]
    t=10 [ 3.90625e-06 -3.90625e-06]
    """

    def __init__(self, *, alpha=2e-3, beta1=0.9, beta2=0.999, **kwds):
        super().__init__(**kwds)
        self._set_param2callable("alpha", alpha)
        self._set_param2callable("beta1", beta1)
        self._set_param2callable("beta2", beta2)

    def __update(self, *, grad=None, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.

        Returns:
            delta (ndarray): Update delta.
        """
        alphakwds = self._get_func_kwds(self.alpha, kwds)
        beta1kwds = self._get_func_kwds(self.beta1, kwds)
        beta2kwds = self._get_func_kwds(self.beta2, kwds)
        alpha_0 = self.alpha(**alphakwds)
        beta1_t = self.beta1(**beta1kwds)
        beta2_t = self.beta2(**beta2kwds)
        self._prod_beta1 *= beta1_t

        self._m += (1-beta1_t)*(grad-self._m)
        self._u = np.maximum(beta2_t*self._u, np.abs(grad))
        alpha_t = alpha_0/(1-self._prod_beta1)
        return -alpha_t*self._m/self._u

    def build(self, *, _m=1e-8, _u=1e-8, **kwds):
        """Build optimizer."""
        self._m = _m
        self._u = _u
        self._prod_beta1 = 1.
        self.update = self.__update
        self._is_built = True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
