import numpy as np
from functools import wraps


def to_numpy(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, list) or isinstance(arg, tuple):
                arg = np.array(arg)
            new_args.append(arg)
        args = tuple(new_args)

        for key, val in kwargs.items():
            if isinstance(val, list) or isinstance(val, tuple):
                kwargs[key] = np.array(val)

        return func(*args, **kwargs)

    return wrapper


def check_box_size(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if np.isclose(sum(self.box), 0.0):
            raise ValueError(
                "Undefined box size, use set_box() before invoking PBC operation!"
            )
        else:
            return func(self, *args, **kwargs)

    return wrapper
