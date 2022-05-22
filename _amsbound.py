import numpy as np

from _base import BaseOpt


class AMSBound(BaseOpt):
    """AMSBound optimizer class.

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
        eta (Callable): Learning rate.
                        If float comming, make lambda function
                        which return the value.
        eta_l (Callable): Lower limit of learning rate.
                          If float comming, make lambda function
                          which return the value.
        eta_u (Callable): Upper limit of learning rate.
                          If float comming, make lambda function
                          which return the value.

    Examples:
    >>> import numpy as np
    >>> obj = AMSBound()
    >>> obj.update(grad=np.array([-0.5, 1])) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: <<Error sentence>>
    >>> obj.build()
    >>> obj.update(grad=np.array([-0.5, 1]))
    array([ 0.00316221, -0.00316226])
    >>> obj = AMSBound(alpha=lambda t=1: 1e-3 * 0.5**(t-1),
    ...                beta1=lambda t=1: 0.9 if t <= 5 else 0.5,
    ...                beta2=lambda t=1: 0.999 if t <= 5 else 0.555,
    ...                eta=lambda t=1: 1e-1 * 0.5**(t-1))
    >>> obj.build()
    >>> for i in range(1, 11):
    ...     print(f"t={i:2d}", obj.update(grad=np.array([-0.5, 1]), t=i))
    t= 1 [ 0.00316221 -0.00316226]
    t= 2 [ 0.00150244 -0.00150245]
    t= 3 [ 0.0007145 -0.0007145]
    t= 4 [ 0.0003401 -0.0003401]
    t= 5 [ 0.00016203 -0.00016203]
    t= 6 [ 0.00032706 -0.00065412]
    t= 7 [ 0.00019053 -0.00038106]
    t= 8 [ 9.98618707e-05 -1.99723742e-04]
    t= 9 [ 5.01736931e-05 -1.00347386e-04]
    t=10 [ 2.47499733e-05 -4.94999466e-05]
    """

    def __init__(self, *, alpha=1e-3, beta1=0.9, beta2=0.999, eta=1e-1,
                 eta_l=lambda eta, beta2, t: eta*(1 - 1/((1-beta2)*t + 1)),
                 eta_u=lambda eta, beta2, t: eta*(1 + 1/((1-beta2)*t)),
                 **kwds):
        super().__init__(**kwds)
        self._set_param2callable("alpha", alpha)
        self._set_param2callable("beta1", beta1)
        self._set_param2callable("beta2", beta2)
        self._set_param2callable("eta", eta)
        self._set_param2callable("eta_l", eta_l)
        self._set_param2callable("eta_u", eta_u)

    def __update(self, *, grad=None, t=1, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.
            t (int): Timestep.

        Returns:
            delta (ndarray): Update delta.
        """
        kwds["t"] = t
        alphakwds = self._get_func_kwds(self.alpha, kwds)
        beta1kwds = self._get_func_kwds(self.beta1, kwds)
        beta2kwds = self._get_func_kwds(self.beta2, kwds)
        etakwds = self._get_func_kwds(self.eta, kwds)
        alpha_t = self.alpha(**alphakwds)
        beta1_t = self.beta1(**beta1kwds)
        beta2_t = self.beta2(**beta2kwds)
        eta_t = self.eta(**etakwds)

        kwds["eta"] = eta_t
        kwds["beta2"] = beta2_t
        etalkwds = self._get_func_kwds(self.eta_l, kwds)
        etaukwds = self._get_func_kwds(self.eta_u, kwds)
        eta_l_t = self.eta_l(**etalkwds)
        eta_u_t = self.eta_u(**etaukwds)

        self._m += (1-beta1_t)*(grad-self._m)
        self._v += (1-beta2_t)*(grad*grad - self._v)
        self._v_hat = np.maximum(self._v_hat, self._v)
        eta_hat = np.clip(alpha_t/np.sqrt(self._v), eta_l_t, eta_u_t)
        return -eta_hat/np.sqrt(t)*self._m

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
