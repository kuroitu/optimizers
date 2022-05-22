import inspect


class BaseOpt():
    """Base class for optimizers."""

    def __init__(self, *args, **kwds):
        self._is_built = False

    def __repr__(self):
        reprstr = self.__class__.__name__ + "(\n\t" \
                + "\n\t".join([argname + "=" + str(argvalue)
                               for argname, argvalue
                               in self.__dict__.items()]) \
                + "\n)"
        return reprstr

    __str__ = __repr__

    def _get_func_param(self, func):
        return inspect.signature(func).parameters

    def _get_func_kwds(self, func, kwds):
        """Get keyword arguments for func.

        Args:
            func (Callable): Target to extract arguments.
            kwds (Dict): Dictionary including func's arguments.
        """
        func_kwds = {}
        params = self._get_func_param(func)
        for param in params.values():
            if param.name in kwds:
                func_kwds[param.name] = kwds[param.name]
        return func_kwds

    def _set_param2callable(self, name, param):
        if callable(param):
            setattr(self, name, param)
        else:
            setattr(self, name, lambda *pargs, **pkwds: param)

    def update(self, *args, **kwds):
        if self._is_built:
            raise NotImplementedError("'update' method must be implemented.")
        else:
            raise RuntimeError(
                    "Not built. "
                    "You must execute 'build' method "
                    "before executing 'update' method.")

    def build(self, *args, **kwds):
        raise NotImplementedError("'build' method must be implemented.")
