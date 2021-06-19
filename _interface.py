import sys
import string
from typing import Any


def get_opt(name: str, *args: Any, **kwds: Any) -> Any:
    """Interface for getting optimizer.

    Args:
        name (str): Target optimizer's name.

    Returns:
        _ (BaseAct): Optimizer class.

    Examples:
    >>> for opt in opt_dict:
    ...     print(get_opt(opt))
    SGD(eta=0.01)
    MSGD(eta=0.01, mu=0.9)
    NAG(parent=None, eta=0.01, mu=0.9)
    AdaGrad(eta=0.001)
    RMSprop(eta=0.01, rho=0.99)
    AdaDelta(rho=0.95)
    Adam(alpha=0.001, beta1=0.9, beta2=0.999)
    RMSpropGraves(eta=0.0001, rho=0.95)
    SMORMS3(eta=0.001)
    AdaMax(alpha=0.002, beta1=0.9, beta2=0.999)
    Nadam(alpha=0.002, mu=0.975, nu=0.999)
    Eve(alpha=0.001, beta1=0.9, beta2=0.999, beta3=0.999, c=10, f_star=0)
    SantaE(eta=0.01, sigma=0.95, anne_rate=0.5, burnin=100, C=5, N=16)
    SantaSSS(eta=0.1, sigma=0.95, anne_rate=0.5, burnin=100, C=5, N=16)
    AMSGrad(alpha=0.001, beta1=0.9, beta2=0.999)
    AdaBound(alpha=0.001, beta1=0.9, beta2=0.999, eta=0.1, eta_l_list=[0.1, 0.999], eta_u_list=[0.1, 0.999])
    AMSBound(alpha=0.001, beta1=0.9, beta2=0.999, eta=0.1, eta_l_list=[0.1, 0.999], eta_u_list=[0.1, 0.999])
    AdaBelief(alpha=0.001, beta1=0.9, beta2=0.999)
    SAM(parent=None, q=2, rho=0.05, opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), opt_dict={})
    """
    from main.dl.opt import opt_dict

    name = name.lower().translate(str.maketrans("", "", string.punctuation))
    return opt_dict[name](*args, **kwds)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
