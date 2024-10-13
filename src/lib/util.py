import os
import time
from typing import Callable

import numpy as np
from numpy._typing import NDArray


def start_timer():
    start = time.time()

    def stop_timer():
        print(f'Time to finish: {time.time() - start:.2f} seconds\n')

    return stop_timer


def get_func_name(func: Callable) -> str:
    return type(func).__name__


def get_class_name(obj) -> str:
    return obj.__class__.__name__


def round(a: NDArray | float, decimals: int = 5):
    return np.round(a, decimals=decimals)


def dir_exist(path: str) -> bool:
    cwd = os.getcwd()
    return os.path.exists(os.path.join(cwd, path))


def create_dir(path: str):
    cwd = os.getcwd()
    os.mkdir(os.path.join(cwd, path))
