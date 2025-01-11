from abc import ABC
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray


class Objective(ABC):
    @property
    @abstractmethod
    def bounds(self) -> NDArray:
        pass

    @abstractmethod
    def evaluate(self, solution: NDArray) -> float:
        pass


class Easom(Objective):
    @property
    def bounds(self):
        return np.array([[-5.0, 5.0], [-5.0, 5.0]])

    def evaluate(self, solution: NDArray):
        x, y = solution
        return -1 * np.cos(x) * np.cos(y) * np.exp(-1 * ((x - np.pi) ** 2 + (y - np.pi) ** 2))


class ThreeHumpCamel(Objective):
    @property
    def bounds(self):
        return np.array([[-5.0, 5.0], [-5.0, 5.0]])

    def evaluate(self, solution: NDArray):
        x, y = solution
        return 2 * x**2 - 1.05 * x**4 + x**6 / 6 + x * y + y**2


class Ackley(Objective):
    @property
    def bounds(self):
        return np.array([[-5.0, 5.0], [-5.0, 5.0]])

    def evaluate(self, solution: NDArray):
        x, y = solution
        return (
            -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
            - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
            + np.e
            + 20
        )


class Sphere(Objective):
    @property
    def bounds(self):
        return np.array([[-5.0, 5.0]])

    def evaluate(self, x: NDArray):
        return x[0] ** 2


class Sphere3D(Objective):
    @property
    def bounds(self):
        return np.array([[-5.0, 5.0] for _ in range(9)])

    def evaluate(self, x: NDArray):
        return np.sum(x**2)
