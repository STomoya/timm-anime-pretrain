from functools import wraps

from storch.distributed.utils import is_primary


def only_on_primary(func):
    @wraps(func)
    def inner(*args, **kwargs):
        if is_primary():
            return func(*args, **kwargs)

    return inner
