import numpy as np

from _base import BaseOpt


class Nadam(BaseOpt):
    """Nadam optimizer class.

    Attributes:
        alpha (Callable): Learning rate.
                          If float comming, make lambda function
                          which return the value.
        mu (Callable): Momentum oblivion rate.
                       If float comming, make lambda function
                       which return the value.
        nu (Callable): Square Momentum oblivion rate.
                       If float comming, make lambda function
                       which return the value.

    Examples:
    >>> import numpy as np
    >>> obj = Nadam()
    >>> obj.update(grad=np.array([-0.5, 1])) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: <<Error sentence>>
    >>> obj.build()
    >>> obj.update(grad=np.array([-0.5, 1]))
    array([ 0.00298878, -0.00298882])
    >>> obj = Nadam(alpha=lambda t=1: 2e-3 * 0.5**(t-1),
    ...             mu=lambda t=1: 0.975 if t <= 5 else 0.575,
    ...             nu=lambda t=1: 0.999 if t <= 5 else 0.555)
    >>> obj.build()
    >>> for i in range(1, 11):
    ...     print(f"t={i:2d}", obj.update(grad=np.array([-0.5, 1]), t=i))
    t= 1 [ 0.00298878 -0.00298882]
    t= 2 [ 0.00116509 -0.0011651 ]
    t= 3 [ 0.00054138 -0.00054139]
    t= 4 [ 0.00026246 -0.00026246]
    t= 5 [ 5.56816642e-05 -5.56818373e-05]
    t= 6 [ 0.00010585 -0.00010585]
    t= 7 [ 4.56883878e-05 -4.56883884e-05]
    t= 8 [ 2.18171150e-05 -2.18171151e-05]
    t= 9 [ 1.07005319e-05 -1.07005319e-05]
    t=10 [ 5.30075911e-06 -5.30075912e-06]
    """

    def __init__(self, *, alpha=2e-3, mu=0.975, nu=0.999, **kwds):
        super().__init__(**kwds)
        self._set_param2callable("alpha", alpha)
        self._set_param2callable("mu", mu)
        self._set_param2callable("nu", nu)

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
        mukwds = self._get_func_kwds(self.mu, kwds)
        nukwds = self._get_func_kwds(self.nu, kwds)
        mu_t = self.mu(**mukwds)
        nu_t = self.nu(**nukwds)
        self._prod_mu *= mu_t
        self._prod_nu *= nu_t
        if "t" in mukwds:
            mukwds["t"] = t + 1
        mu_tp1 = self.mu(**mukwds)

        self._m += (1-mu_t)*(grad-self._m)
        self._v += (1-nu_t)*(grad*grad - self._v)
        m_hat = self._m*mu_t/(1 - self._prod_mu*mu_tp1) \
              + grad*(1-mu_t)/(1-self._prod_mu)
        v_hat = self._v*nu_t/(1-self._prod_nu)
        return -self.alpha(**alphakwds)*m_hat/np.sqrt(v_hat)

    def build(self, *, _m=1e-8, _v=1e-8, **kwds):
        """Build optimizer."""
        self._m = _m
        self._v = _v
        self._prod_mu = 1.
        self._prod_nu = 1.
        self.update = self.__update
        self._is_built = True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
