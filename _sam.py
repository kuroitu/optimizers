import numpy as np

from _base import BaseOpt


class SAM(BaseOpt):
    """SAM optimizer class.

    Attributes:
        q (Callable): The value that determines the norm.
                      If integer comming, make lambda function
                      which return the value.
        rho (Callable): Neighborhood size.
                        If float comming, make lambda function
                        which return the value.
        opt (BaseOpt): Optimizer to update delta.

    Examples:
    >>> import numpy as np
    >>> from _sgd import SGD
    >>> opt = SGD()
    >>> opt.build()
    >>> class SAMTest():
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
    >>> parent = SAMTest(x=np.array([-0.5, 1]))
    >>> obj = SAM()
    >>> obj.update(grad=np.array([-0.5, 1]),
    ...            parent=parent) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: <<Error sentence>>
    >>> obj.build(opt=opt)
    >>> obj.update(grad=np.array([-0.5, 1]), parent=parent)
    array([-0.00523644, -0.02094574])
    >>> obj = SAM(rho=lambda t=1: 5e-2 * 0.5**(t-1))
    >>> obj.build(opt=opt)
    >>> for i in range(1, 11):
    ...     print(f"t={i:2d}", obj.update(grad=np.array([-0.5, 1]),
    ...                                   parent=parent, t=i))
    t= 1 [-0.00523644 -0.02094574]
    t= 2 [-0.00511822 -0.02047287]
    t= 3 [-0.00505911 -0.02023644]
    t= 4 [-0.00502955 -0.02011822]
    t= 5 [-0.00501478 -0.02005911]
    t= 6 [-0.00500739 -0.02002955]
    t= 7 [-0.00500369 -0.02001478]
    t= 8 [-0.00500185 -0.02000739]
    t= 9 [-0.00500092 -0.02000369]
    t=10 [-0.00500046 -0.02000185]
    """

    def __init__(self, *, q=2, rho=5e-2, **kwds):
        super().__init__(**kwds)
        self._set_param2callable("q", q)
        self._set_param2callable("rho", rho)

    def __update(self, *, grad=None, parent=None, **kwds):
        """Update calculation.

        Args:
            grad (ndarray): Gradient propagating from the lower layer.
            parent (BaseLayer): Functions that this optimizer
                                tries to optimize.

        Returns:
            delta (ndarray): Update delta.
        """
        eps = self._epsilon(grad, **kwds)

        # Repropagation.
        parent.params += eps
        _ = parent.forward(**kwds)
        grad = parent.backward(grad=grad, **kwds)
        parent.params -= eps

        return self.opt.update(grad=grad, **kwds)

    def _epsilon(self, grad, **kwds):
        kwds["grad"] = grad
        rhokwds = self._get_func_kwds(self.rho, kwds)
        qkwds = self._get_func_kwds(self.q, kwds)
        rho_t = self.rho(**rhokwds)
        q_t = self.q(**qkwds)
        return (rho_t*np.sign(grad)
                * (np.abs(grad)**(q_t-1))
                  /np.linalg.norm(grad, ord=q_t)**(1 - 1/q_t))

    def build(self, *, opt=None, **kwds):
        """Build optimizer.

        Args:
            opt (BaseOpt): Optimizer to compute update delta.
        """
        if opt is None:
            raise ValueError("'SAM' require 'opt'.")
        self.opt = opt
        self.update = self.__update
        self._is_built = True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
