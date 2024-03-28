import functools
from time import perf_counter

def timed(func):
    def wrapper(*args, **kwargs):
        t_initial = perf_counter()
        output = func(*args, **kwargs)
        t_final = perf_counter()
        print(f'Finished {func.__name__} in {t_final - t_initial} seconds')
        return output
    return wrapper

