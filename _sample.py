from typing import Union
from dataclasses import dataclass

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import matplotlib.animation as anim

try:
    from ._interface import get_opt
except ImportError:
    # For doctest
    from main.dl.opt import get_opt


@dataclass
class _target():
    params: Union[ndarray, int] = 0

    def forward(self, x, *args, **kwds):
        """
        Maxima = -0.257043 (when x = -0.0383009)
        Minimal = -2.36643 (when x = 2.22366)
                = -2.93536 (when x = -2.93536)
        """
        if np.any(self.params!=x):
            x = self.params
        return -np.exp(-((x+4)*(x+1)*(x-1)*(x-3)/14 + 0.5))


    def backward(self, x, *args, **kwds):
        """
        https://www.wolframalpha.com/input/?i=-exp%28-%28%28x+%2B+4%29%28x+%2B+1%29%28x+%E2%88%92+1%29%28x+%E2%88%92+3%29%2F14+%2B+0.5%29%29&lang=ja
        """
        if np.any(self.params!=x):
            x = self.params
        return (0.173294*np.exp(-((x+4)*(x+1)*(x-1)*(x-3)/14))
                        *(x**3 + 0.75 * x**2 - 6.5*x - 0.25))


def opt_plot():
    objective = _target()

    start_x = objective.params = 3.5
    start_y = objective.forward(start_x)

    x_range = objective.params = np.arange(-5, 5, 1e-2)
    y_range = objective.forward(x_range)

    exact_x = objective.params = -2.93536
    exact_y = objective.forward(exact_x)

    semi_exact_x = objective.params = 2.22366
    semi_exact_y = objective.forward(semi_exact_x)

    epoch = 256
    frame = 64
    fps = 10

    np.random.seed(seed=2)
    opt_dict = {"SGD": get_opt("sgd", n=1, eta=0.1),
                "MSGD": get_opt("msgd", n=1, eta=0.25),
                "NAG": get_opt("nag", n=1, parent=objective, eta=0.1),
                "AdaGrad": get_opt("adagrad", n=1, eta=0.25),
                "RMSprop": get_opt("rmsprop", n=1, eta=0.05),
                "AdaDelta": get_opt("adadelta", n=1, rho=0.9999),
                "Adam": get_opt("adam", n=1, alpha=0.25),
                "RMSpropGraves": get_opt("rmspropgraves", n=1, eta=0.0125),
                "SMORMS3": get_opt("smorms3", n=1, eta=0.05),
                "AdaMax": get_opt("adamax", n=1, alpha=0.5),
                "Nadam": get_opt("nadam", n=1, alpha=0.5),
                "Eve": get_opt("eve", n=1, f_star=exact_y, alpha=0.25),
                "SantaE": get_opt("santae", n=1, burnin=epoch/2**4, N=1,
                                  eta=0.0125),
                "SantaSSS": get_opt("santasss", n=1, burnin=epoch/2**4, N=1,
                                    eta=0.125),
                "AMSGrad": get_opt("amsgrad", n=1, alpha=0.125),
                "AdaBound": get_opt("adabound", n=1, alpha=0.125),
                "AMSBound": get_opt("amsbound", n=1, alpha=0.125),
                "AdaBelief": get_opt("adabelief", n=1, alpha=0.25),
                "SAM": get_opt("sam", n=1, parent=objective,
                               opt_dict={"n": 1, "alpha": 0.25})
                }
    key_len = len(max(opt_dict.keys(), key=len))

    current_x = np.full(len(opt_dict), start_x)
    current_y = np.full(len(opt_dict), start_y)
    err_list = [[] for i in range(len(opt_dict))]
    semi_err_list = [[] for i in range(len(opt_dict))]

    cmap = plt.get_cmap("rainbow")
    coloring = [cmap(i) for i in np.linspace(0, 1, len(opt_dict))]
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), dpi=60)
    fig.suptitle("Optimizer comparison")
    ax[0].set_title("optimize visualization")
    ax[0].set_position([0.075, 0.6, 0.75, 0.3])
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].grid()
    ax[0].set_xlim([x_range[0], x_range[-1]])
    ax[0].set_ylim([np.min(y_range)-1, np.max(y_range)+1])

    ax[1].set_title("error from minimum")
    ax[1].set_position([0.075, 0.325, 0.75, 0.2])
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("error")
    ax[1].set_yscale("log")
    ax[1].grid()
    ax[1].set_xlim([0, epoch])
    ax[1].set_ylim([1e-16, 25])

    ax[2].set_title("error from semi-minimum")
    ax[2].set_position([0.075, 0.05, 0.75, 0.2])
    ax[2].set_xlabel("epoch")
    ax[2].set_ylabel("error")
    ax[2].set_yscale("log")
    ax[2].grid()
    ax[2].set_xlim([0, epoch])
    ax[2].set_ylim([1e-16, 25])

    base = ax[0].plot(x_range, y_range, color="b")

    images = []
    for i in range(1, epoch+1):
        imgs = []
        for j, opt in enumerate(opt_dict):
            err = 0.5 * (current_x[j] - exact_x)**2
            semi_err = 0.5 * (current_x[j] - semi_exact_x)**2
            err_list[j].append(err)
            semi_err_list[j].append(semi_err)

            if i % (epoch//frame) == 0:
                img, = ax[0].plot(current_x[j], current_y[j],
                                  color=coloring[j], marker="o")
                err_img, = ax[1].plot(err_list[j], color=coloring[j])
                err_point, = ax[1].plot(i-1, err, color=coloring[j],
                                        marker="o")
                semi_err_img, = ax[2].plot(semi_err_list[j], color=coloring[j])
                semi_err_point, = ax[2].plot(i-1, semi_err, color=coloring[j],
                                             marker="o")
                imgs.extend([img, err_img, err_point,
                             semi_err_img, semi_err_point])

            objective.params = current_x[j]
            dw = opt_dict[opt].update(objective.backward(current_x[j]),
                                      current_x[j],
                                      t=i, f=current_y[j])
            current_x[j] += dw
            objective.params = current_x[j]
            current_y[j] = objective.forward(current_x[j])
        if imgs:
            images.append(base+imgs)
    imgs = []
    for j, opt in enumerate(opt_dict):
        img, = ax[0].plot(current_x[j], current_y[j],
                          marker="o", color=coloring[j], label=opt)
        imgs.append(img)
        print(f"{opt.rjust(key_len)} method get {current_x[j]}.")

        err = 0.5 * (current_x[j] - exact_x)**2
        semi_err = 0.5 * (current_x[j] - semi_exact_x)**2
        err_list[j].append(err)
        semi_err_list[j].append(semi_err)

        img, = ax[1].plot(err_list[j], color=coloring[j])
        imgs.append(img)
        img, = ax[1].plot(epoch, err, color=coloring[j], marker="o")
        imgs.append(img)

        img, = ax[2].plot(semi_err_list[j], color=coloring[j])
        imgs.append(img)
        img, = ax[2].plot(i-1, semi_err, color=coloring[j], marker="o")
        imgs.append(img)
    images.append(base+imgs)
    fig.legend(bbox_to_anchor=(1., 0.85))

    ani = anim.ArtistAnimation(fig, images)
    ani.save("optimizers.gif", writer="pillow", fps=fps)

if __name__ == "__main__":
    opt_plot()
