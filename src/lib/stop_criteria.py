from typing import List

import numpy as np

from src.lib.common import Individ


def euclidean_distance(point_a, point_b):
    return np.sqrt(np.sum((point_a - point_b) ** 2))


def has_close_fitnesses(children: List[Individ], parents: List[Individ], e: float):
    parents_mean = np.mean([individ.fitness for individ in parents])
    children_mean = np.mean([individ.fitness for individ in children])
    diff = np.abs(parents_mean - children_mean)
    return diff < e


def has_close_solutions(children: List[Individ], parents: List[Individ], e: float,) -> bool:
    n = len(children)
    max_distance = -1
    for i in range(n):
        for j in range(i+1, n):
            distance = euclidean_distance(
                children[i].solutions,
                children[j].solutions
            )
            if distance > max_distance:
                max_distance = distance

    return max_distance < e


def has_close_solutions_nd(children: List[Individ], parents: List[Individ], e: float,) -> bool:
    n = len(children)
    max_distance = -1
    for i in range(n):
        for j in range(i+1, n):
            distance = euclidean_distance(
                children[i].solutions,
                children[j].solutions
            )
            if distance > max_distance:
                max_distance = distance

    return max_distance < e
