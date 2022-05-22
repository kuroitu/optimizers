import numpy as np

from _base import BaseOpt


class AMSGrad(BaseOpt):
    """AMSGrad optimizer class.

    Attributes:
        alpha (Callable): Learning rate.
                          If float comming, make lambda function
                          which return the value.
        beta1 (Callable): Momentum oblivion rate.
                          If float comming, make lambda function
                          which return the value.
        beta2 (Callable): Square Momentum oblivion rate.
                          If float comming, make lambda function
                          which return the value.

    Examples:
    >>> import numpy as np
    >>> obj = AMSGrad()
    >>> obj.update(grad=np.array([-0.5, 1])) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: <<Error sentence>>
    >>> obj.build()
    >>> obj.update(grad=np.array([-0.5, 1]))
    array([ 0.00316221, -0.00316226])
    >>> obj = AMSGrad(alpha=lambda t=1: 1e-3 * 0.5**(t-1),
    ...               beta1=lambda t=1: 0.9 if t <= 5 else 0.5,
    ...               beta2=lambda t=1: 0.999 if t <= 5 else 0.555)
    >>> obj.build()
    >>> for i in range(1, 11):
    ...     print(f"t={i:2d}", obj.update(grad=np.array([-0.5, 1]), t=i))
    t= 1 [ 0.00316221 -0.00316226]
    t= 2 [ 0.00212477 -0.00212479]
    t= 3 [ 0.00123755 -0.00123756]
    t= 4 [ 0.0006802 -0.0006802]
    t= 5 [ 0.00036232 -0.00036232]
    t= 6 [ 3.29125052e-05 -3.29125062e-05]
    t= 7 [ 1.59928179e-05 -1.59928181e-05]
    t= 8 [ 7.94285912e-06 -7.94285917e-06]
    t= 9 [ 3.95332119e-06 -3.95332120e-06]
    t=10 [ 1.96937137e-06 -1.96937137e-06]
    """

    def __init__(self, *, alpha=1e-3, beta1=0.9, beta2=0.999, **kwds):
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
        alpha_t = self.alpha(**alphakwds)
        beta1_t = self.beta1(**beta1kwds)
        beta2_t = self.beta2(**beta2kwds)

        self._m += (1-beta1_t)*(grad-self._m)
        self._v += (1-beta2_t)*(grad*grad - self._v)
        self._v_hat = np.maximum(self._v_hat, self._v)
        return -alpha_t*self._m/np.sqrt(self._v_hat)

    def build(self, *, _m=1e-8, _v=1e-8, _v_hat=1e-8, **kwds):
        """Build optimizer."""
        self._m = _m
        self._v = _v
        self._v_hat = _v_hat
        self.update = self.__update
        self._is_built = True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
