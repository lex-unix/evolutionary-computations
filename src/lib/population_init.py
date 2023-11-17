from typing import List

from numpy import random
from numpy._typing import NDArray

from src.lib.common import Individ


def uniform_population(size: int, bounds: NDArray) -> List[Individ]:
    pop = random.uniform(bounds[:, 0], bounds[:, 1], (size, len(bounds)))
    return [Individ(x) for x in pop]


def distrubited_population(size: int, bounds: NDArray) -> List[Individ]:
    dims = len(bounds)
    population = [
        Individ(bounds[:, 0] + random.rand(dims) * (bounds[:, 1] - bounds[:, 0]))  # nopep8
        for _ in range(size)
    ]
    return population
