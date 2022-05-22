from typing import Union
from dataclasses import dataclass

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from _sgd import SGD
from _msgd import MSGD
from _nag import NAG
from _adagrad import AdaGrad
from _rmsprop import RMSprop
from _adadelta import AdaDelta
from _adam import Adam
from _rmsprop_graves import RMSpropGraves
from _smorms3 import SMORMS3
from _adamax import AdaMax
from _nadam import Nadam
from _eve import Eve
from _santa_e import SantaE
from _santa_sss import SantaSSS
from _amsgrad import AMSGrad
from _adabound import AdaBound
from _amsbound import AMSBound
from _adabelief import AdaBelief
from _sam import SAM


@dataclass
class _target():
    params: Union[ndarray, int] = 0

    def forward(self, *, x=None, **kwds):
        """
        Maxima  = -0.256097 (when x = -0.0704646)
        Minimal = -0.915094 (when x =  1.57421)
                = -3.85712  (when x = -2.25375)
        """
        if x is None:
            x = self.params
        return -np.exp(-((x+3)*(x+1)*(x-1)*(x-2)/7 + 0.5))

    def backward(self, *, x=None, **kwds):
        """
        https://www.wolframalpha.com/input?i=d%2Fdx+-exp%28-%28%28x%2B3%29*%28x%2B1%29*%28x-1%29*%28x-2%29%2F7%2B0.5%29%29&lang=ja
        """
        if x is None:
            x = self.params
        return (0.346589*np.exp(-((x-2)*(x-1)*(x+1)*(x+3)/7))
                        *(x-1.57421)*(x+0.0704646)*(x+2.25375))


def opt_plot():
    objective = _target()

    start_x = objective.params = np.array([2.5])
    start_y = objective.forward(x=start_x)

    x_range = objective.params = np.arange(-3, 3, 1e-2)
    y_range = objective.forward(x=x_range)

    exact_x = objective.params = np.array([-2.25375])
    exact_y = objective.forward(x=exact_x)

    semi_exact_x = objective.params = np.array([1.57421])
    semi_exact_y = objective.forward(x=semi_exact_x)

    epoch = 2**8
    frame = 2**6
    fps = 10
    seed = 2

    opt_dict = {
            "SGD": SGD(),
            "MSGD": MSGD(),
            "NAG": NAG(),
            "AdaGrad": AdaGrad(eta=5e-2),
            "RMSprop": RMSprop(),
            "AdaDelta": AdaDelta(rho=0.999),
            "Adam": Adam(alpha=5e-2),
            "RMSpropGraves": RMSpropGraves(eta=5e-3),
            "SMORMS3": SMORMS3(eta=1e-2),
            "AdaMax": AdaMax(alpha=1e-2),
            "Nadam": Nadam(alpha=1e-2),
            "Eve": Eve(f_star=exact_y, alpha=5e-2),
            "SantaE": SantaE(n=1),
            "SantaSSS": SantaSSS(n=1),
            "AMSGrad": AMSGrad(alpha=5e-2),
            "AdaBound": AdaBound(alpha=5e-2),
            "AMSBound": AMSBound(alpha=5e-2),
            "AdaBelief": AdaBelief(),
            "SAM": SAM(),
    }
    key_len = len(max(opt_dict.keys(), key=len))
    sam_opt = SGD()
    sam_opt.build()

    current_x = np.full(len(opt_dict), start_x).reshape(-1, 1)
    current_y = np.full(len(opt_dict), start_y).reshape(-1, 1)
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
    #ax[1].set_ylim([1e-16, 25])

    ax[2].set_title("error from semi-minimum")
    ax[2].set_position([0.075, 0.05, 0.75, 0.2])
    ax[2].set_xlabel("epoch")
    ax[2].set_ylabel("error")
    ax[2].set_yscale("log")
    ax[2].grid()
    ax[2].set_xlim([0, epoch])
    #ax[2].set_ylim([1e-16, 25])

    base = ax[0].plot(x_range, y_range, color="b")

    for opt in opt_dict.values():
        opt.build(shape=(1,), seed=seed, opt=sam_opt)
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
            dw = opt_dict[opt].update(grad=objective.backward(x=current_x[j]),
                                      parent=objective,
                                      t=i, f=current_y[j])
            current_x[j] += dw
            objective.params = current_x[j]
            current_y[j] = objective.forward(x=current_x[j])
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
