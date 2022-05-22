from dataclasses import dataclass

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D

try:
    from ._interface import get_opt
except ImportError:
    # For doctest
    from main.dl.opt import get_opt


@dataclass
class _target():
    test_type: int = 3

    def __post_init__(self, *args, **kwds):
        if self.test_type == 1:
            self.params = np.array([1e-4, 4.])
            self.exact = np.array([5., 0.])
            self.elevation = 1.
            self.view_init = (35,)
            self.epoch = 2**7
            self.seed = 0
        elif self.test_type == 2:
            self.params = np.array([-1., 2.])
            self.exact = np.array([0., 0.])
            self.elevation = 0.25
            self.view_init = (75,)
            self.epoch = 2**7
            self.seed = 0
        elif self.test_type == 3:
            self.params = np.array([-3., 4.])
            self.exact = np.array([0., 0.])
            self.elevation = 0.125
            self.view_init = (55,)
            self.epoch = 2**10
            self.seed = 543
        elif self.test_type == 4:
            self.params = np.array([-2., 2.])
            self.exact = np.array([0., 0.])
            self.elevation = 0.25
            self.view_init = (45, -87)
            self.epoch = 2**8
            self.seed = 3
#            self.params = np.array([-0.5, 0.])
#            self.elevation = 0.125
#            self.view_init = (25, -87)

    def forward(self, x, *args, **kwds):
        if np.any(self.params!=x):
            x, y = self.params
        else:
            x, y = x
        if self.test_type == 1:
            return y**2 - x**2
        elif self.test_type == 2:
            return np.tanh(x)**2 + np.tanh(y)**2
        elif self.test_type == 3:
            return (-(np.sinc(x)+np.sinc(y)) + (x**2 + y**2)/10)
        elif self.test_type == 4:
            return (0.125*(x**2 + y**2) + np.tanh(x*10)**2)
#            return (0.125*(0.5*x**2 + 0.125*y**2) + np.tanh(x*10)**2)

    def backward(self, x, *args, **kwds):
        if np.any(self.params!=x):
            x, y = self.params
        else:
            x, y = x
        if self.test_type == 1:
            dw = -2*x
            db = 2*y
        elif self.test_type == 2:
            dw = 2 * np.tanh(x) / np.cosh(x)**2
            db = 2 * np.tanh(y) / np.cosh(y)**2
        elif self.test_type == 3:
            dw = (np.sin(np.pi*x)/(np.pi * x**2) + 2*x/10 - np.cos(np.pi*x)/x)
            db = (np.sin(np.pi*y)/(np.pi * y**2) + 2*y/10 - np.cos(np.pi*y)/y)
        elif self.test_type == 4:
            dw = (0.25*x + 20 * np.tanh(x*10) / np.cosh(x*10)**2)
            db = 0.25*y
#            dw = (0.5*0.25*x + 20 * np.tanh(x*10) / np.cosh(x*10)**2)
#            db = 0.125*0.25*y
        return np.array([dw, db])

    def get_exact(self, *args, **kwds):
        params = self.params
        self.params = self.exact
        exact_z = self.forward(self.exact)
        self.params = params
        return exact_z


class TrajectoryAnimation3D(anim.FuncAnimation):
    def __init__(self, paths, labels=[], fig=None, ax=None,
                 blit=True, coloring=None, **kwargs):
        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
        else:
            if ax is None:
                ax = fig.gca()

        self.fig = fig
        self.ax = ax
        self.paths = paths

        frames = paths.shape[0]

        self.lines = []
        self.points = []
        for j, opt in enumerate(labels):
            line, = ax.plot([], [], [], label=opt, lw=2, color=coloring[j])
            point, = ax.plot([], [], [], marker="o", color=coloring[j])
            self.lines.append(line)
            self.points.append(point)

        super().__init__(fig, self.animate,
                         frames=frames, blit=blit, **kwargs)

    def animate(self, i):
        start = 0 if i-8 < 0 else i-8
        j = 0
        for line, point in zip(self.lines, self.points):
            line.set_data(self.paths[start:i+1, j, 0],
                          self.paths[start:i+1, j, 1])
            line.set_3d_properties(self.paths[start:i+1, j, 2])
            line.set_zorder(i+100)
            point.set_data(self.paths[i, j, 0], self.paths[i, j, 1])
            point.set_3d_properties(self.paths[i, j, 2])
            point.set_zorder(i+101)
            j += 1
        return self.lines + self.points


def opt_plot():
    objective = _target(test_type=4)
    start_x, start_y = objective.params
    start_z = objective.forward([start_x, start_y])

    x_range = np.arange(-5, 5, 1e-2)
#    x_range = np.arange(-2, 2, 1e-2)
    y_range = np.arange(-5, 5, 1e-2)
    X, Y = np.meshgrid(x_range, y_range)
    objective.params = np.array([X, Y])
    Z = objective.forward([X, Y])
    elevation = np.arange(np.min(Z), np.max(Z), objective.elevation)

    exact_z = objective.get_exact()

    epoch = objective.epoch
    frames = 2**6
#    frames = 2**7
    fps = 10

    np.random.seed(seed=objective.seed)
    opt_dict = {"SGD": get_opt("sgd", eta=0.0875),
                "MSGD": get_opt("msgd", eta=0.1),
                "NAG": get_opt("nag", parent=objective, eta=0.1),
                "AdaGrad": get_opt("adagrad", eta=0.25),
                "RMSprop": get_opt("rmsprop", eta=0.05),
                "AdaDelta": get_opt("adadelta", rho=0.9999),
                "Adam": get_opt("adam", alpha=0.25),
                "RMSpropGraves": get_opt("rmspropgraves", eta=0.0125),
                "SMORMS3": get_opt("smorms3", eta=0.05),
                "AdaMax": get_opt("adamax", alpha=0.5),
                "Nadam": get_opt("nadam", alpha=0.5),
                "Eve": get_opt("eve", f_star=exact_z, alpha=0.25),
                "SantaE": get_opt("santae", burnin=epoch/2**3, N=1,
                                  eta=0.0125),
                "SantaSSS": get_opt("santasss", burnin=epoch/2**3, N=1,
                                    eta=0.0125),
                "AMSGrad": get_opt("amsgrad", alpha=0.125),
                "AdaBound": get_opt("adabound", alpha=0.125),
                "AMSBound": get_opt("amsbound", alpha=0.125),
                "AdaBelief": get_opt("adabelief", alpha=0.25),
                "SAM": get_opt("sam", parent=objective,
                               opt_dict={"alpha": 0.25})
                }
#    opt_dict["SGD"] = get_opt("sgd", eta=0.05)
#    opt_dict["RMSprop"] = get_opt("rmsprop", eta=0.05)
#    opt_dict["Adam"] = get_opt("adam", alpha=0.005)
#    opt_dict["AdaBelief"] = get_opt("adabelief", alpha=0.005)
#    opt_dict["SAM"] = get_opt("sam", parent=objective,
#                              opt_dict={"alpha": 0.25})
    key_len = len(max(opt_dict.keys(), key=len))

    current_x = np.full(len(opt_dict), start_x)
    current_y = np.full(len(opt_dict), start_y)
    current_z = np.full(len(opt_dict), start_z)
    paths = np.zeros((frames, len(opt_dict), 3))

    cmap = plt.get_cmap("rainbow")
    coloring = [cmap(i) for i in np.linspace(0, 1, len(opt_dict))]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("Optimizer comparison")
    ax.set_title("optimize visualization")
    ax.set_position([0., 0.1, 0.7, 0.8])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid()
    ax.set_xlim([x_range[0], y_range[-1]])
#    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([y_range[0], y_range[-1]])
    ax.set_zlim([elevation[0], elevation[-1]])
    ax.view_init(*objective.view_init)
    ax.plot_surface(X, Y, Z, cmap="coolwarm", zorder=-10, alpha=0.75)
    ax.contour(X, Y, Z, cmap="autumn", levels=elevation, zorder=-5, alpha=0.75)

    for i in range(1, epoch+1):
        for j, opt in enumerate(opt_dict):
            if (i-1) % (epoch//frames) == 0:
                paths[(i-1)//(epoch//frames), j, 0] = current_x[j]
                paths[(i-1)//(epoch//frames), j, 1] = current_y[j]
                paths[(i-1)//(epoch//frames), j, 2] = current_z[j]

            objective.params = np.array([current_x[j], current_y[j]])
            dx, dy = opt_dict[opt].update(
                    objective.backward(current_x[j], current_y[j]),
                    np.array(current_x[j], current_y[j]),
                    t=i, f=objective.forward([current_x[j], current_y[j]]))
            current_x[j] += dx
            current_y[j] += dy
            objective.params = np.array([current_x[j], current_y[j]])
            current_z[j] = objective.forward([current_x[j], current_y[j]])

    ani = TrajectoryAnimation3D(paths, labels=opt_dict, fig=fig, ax=ax,
                                coloring=coloring)
    fig.legend(bbox_to_anchor=(1., 0.85))
    ani.save("optimizers_3d.gif", writer="pillow", fps=fps)


if __name__ == "__main__":
    opt_plot()
