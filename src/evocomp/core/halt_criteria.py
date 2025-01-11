from abc import ABC
from abc import abstractmethod

import numpy as np

from evocomp.core.candidate import Candidate


class HaltCriteria(ABC):
    def __init__(self, e: float) -> None:
        if e <= 0:
            raise ValueError('epsilon cannot be zero or negative')
        self.e = e

    def euclidean_distance(self, point_a, point_b):
        return np.sqrt(np.sum((point_a - point_b) ** 2))

    @abstractmethod
    def halt(self, children: list[Candidate], parents: list[Candidate]) -> bool:
        pass


class FitnessConvergence(HaltCriteria):
    def __init__(self, e: float) -> None:
        super().__init__(e)

    def halt(self, children: list[Candidate], parents: list[Candidate]) -> bool:
        children_mean = np.mean([candidate.fitness for candidate in children])
        parents_mean = np.mean([candidate.fitness for candidate in parents])
        diff = np.abs(parents_mean - children_mean)
        return diff < self.e


class SolutionConvergence(HaltCriteria):
    def __init__(self, e: float) -> None:
        super().__init__(e)

    def halt(self, children: list[Candidate], parents: list[Candidate]) -> bool:
        n = len(children)
        max_distance = -1
        for i in range(n):
            for j in range(i + 1, n):
                distance = self.euclidean_distance(children[i].solution, children[j].solution)
                if distance > max_distance:
                    max_distance = distance
        return max_distance < self.e
