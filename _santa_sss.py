import numpy as np

from _base import BaseOpt


class SantaSSS(BaseOpt):
    """SantaSSS optimizer class.

     Attributes:
        eta (Callable): Learning rate.
                        If float comming, make lambda function
                        which return the value.
        sigma (Callable): Square momentum oblivion rate.
                          If float comming, make lambda function
                          which return the value.
        anne_func (Callable): Annealing function.
                              This must be to be infinity
                              when timestep goes to infinity.
                              Arguments:
                                t (int): Timestep
                                rate (float): Annealing rate.
                                coef (float): Annealing coef.
                                bias (float): Annealing bias.
        anne_rate (Callable): Annealing rate.
                              If float comming, make lambda function
                              which return the value.
        anne_coef (Callable): Annealing coefficient.
                              If float comming, make lambda function
                              which return the value.
        anne_bias (Callable): Annealing bias.
                              If float comming, make lambda function
                              which return the value.
        burnin (int): Timestep to finish annealing.
        n (Callable): Square momentum relaxation term
                      which is usually the size of (mini)batch.
                      If int commint, make lambda function
                      which return the value.

   Examples:
    >>> import numpy as np
    >>> obj = SantaSSS()
    >>> obj.update(grad=np.array([-0.5, 1])) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: <<Error sentence>>
    >>> obj.build(seed=42, shape=(2,))
    >>> obj.update(grad=np.array([-0.5, 1]))
    array([ 4.55376169, -6.38437135])
    >>> obj = SantaSSS(eta=lambda t=1: 1e-3 * 0.5**(t-1),
    ...                sigma=lambda t=1: 0.95 if t <= 5 else 0.55)
    >>> obj.build(seed=42, shape=(2,))
    >>> for i in range(1, 11):
    ...     print(f"t={i:2d}", obj.update(grad=np.array([-0.5, 1]), t=i))
    t= 1 [ 0.55916603 -0.96134094]
    t= 2 [ 0.63562509 -1.51586982]
    t= 3 [ 0.77185382 -1.06714157]
    t= 4 [ 0.88791285 -0.64682316]
    t= 5 [ 0.60906431 -0.45709684]
    t= 6 [ 0.38072127 -0.34052298]
    t= 7 [ 0.26077463 -0.29131578]
    t= 8 [ 0.15242967 -0.21753294]
    t= 9 [ 0.09895932 -0.16648416]
    t=10 [ 0.08171564 -0.13292416]
    """

    def __init__(self, *,
                 eta=1e-1, sigma=0.95,
                 anne_func=lambda t, rate, coef, bias: coef * t**rate + bias,
                 anne_rate=0.5, anne_coef=1., anne_bias=0.,
                 burnin=100, n=16, **kwds):
        annefuncparam = self._get_func_param(anne_func)
        if "t" not in annefuncparam and "rate" not in annefuncparam \
                and "coef" not in annefuncparam and "bias" not in annefuncparam:
            raise ValueError("'anne_func' must have parameters;\n"
                             "'t', 'rate', 'coef', 'bias'.")

        super().__init__(**kwds)
        self._set_param2callable("eta", eta)
        self._set_param2callable("sigma", sigma)
        self._set_param2callable("anne_rate", anne_rate)
        self._set_param2callable("anne_coef", anne_coef)
        self._set_param2callable("anne_bias", anne_bias)
        self._set_param2callable("n", n)
        self.anne_func = anne_func
        self.burnin = burnin

    def __update(self, *, grad=None, t=1, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.
            t (int): Timestep.

        Returns:
            delta (ndarray): Update delta.
        """
        kwds["t"] = t
        etakwds = self._get_func_kwds(self.eta, kwds)
        sigmakwds = self._get_func_kwds(self.sigma, kwds)
        anneratekwds = self._get_func_kwds(self.anne_rate, kwds)
        annecoefkwds = self._get_func_kwds(self.anne_coef, kwds)
        annebiaskwds = self._get_func_kwds(self.anne_bias, kwds)
        nkwds = self._get_func_kwds(self.n, kwds)
        eta_t = self.eta(**etakwds)
        sigma_t = self.sigma(**sigmakwds)
        annerate_t = self.anne_rate(**anneratekwds)
        annecoef_t = self.anne_coef(**annecoefkwds)
        annebias_t = self.anne_bias(**annebiaskwds)
        beta_t = self.anne_func(t, annerate_t, annecoef_t, annebias_t)
        n_t = self.n(**nkwds)

        v_t = self._v + (1-sigma_t)*(grad * grad / n_t**2 - self._v)
        g_t = 1 / v_t**0.25
        eta_div_beta = eta_t/beta_t
        delta = 0.5*g_t*self._u
        if t < self.burnin:
            alpha_t = self._alpha + 0.5*(self._u*self._u - eta_div_beta)
            u_t = np.exp(-0.5*alpha_t)*self._u
            u_t -= eta_t*g_t*grad
            u_t += (np.sqrt(2*eta_div_beta*self._g)
                    * np.random.randn(*self._u.shape))
            u_t += eta_div_beta*(1 - self._g/g_t)/self._u
            u_t *= np.exp(-0.5*alpha_t)
            alpha_t += 0.5*(u_t*u_t - eta_div_beta)
        else:
            alpha_t = self._alpha
            u_t = np.exp(-0.5*alpha_t)*self._u
            u_t -= eta_t*g_t*grad
            u_t *= np.exp(-0.5*alpha_t)
        delta += 0.5*g_t*u_t

        # Update timestep of t-1 and t.
        self._alpha = alpha_t
        self._v = v_t
        self._g = g_t
        self._u = u_t
        return delta

    def build(self, *,
              c=5, seed=None, shape=None,
              _v=1e-8, _g=1e-8, **kwds):
        """Build optimizer."""
        if shape is None:
            raise ValueError("'SantaE' require 'shape'.")

        np.random.seed(seed)
        etakwds = self._get_func_kwds(self.eta, kwds)
        eta_0 = self.eta(**etakwds)
        self._alpha = eta_0**0.5 * c
        self._u = eta_0**0.5 * np.random.randn(*shape)
        self._v = _v
        self._g = _g
        self.update = self.__update
        self._is_built = True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
