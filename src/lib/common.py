import numpy as np


class Individ:

    def __init__(self, solutions):
        self.solutions = solutions
        self.fitness = None


def calculate_similarity(individ1: Individ, individ2: Individ) -> float:
    dx = individ1.solutions[0] - individ2.solutions[0]
    dy = individ1.solutions[1] - individ2.solutions[1]
    return np.sqrt(dx * dx + dy * dy)


def has_close_fitnesses(children: list[Individ], parents: list[Individ], e: float):
    parents_mean = np.mean([individ.fitness for individ in parents])
    children_mean = np.mean([individ.fitness for individ in children])
    diff = np.abs(parents_mean - children_mean)
    return diff < e


def has_close_solutions(children: list[Individ], parents, e: float,):
    n = len(children)
    max_distance = -1
    for i in range(n):
        for j in range(i+1, n):
            distance = calculate_similarity(children[i], children[j])
            if distance > max_distance:
                max_distance = distance

    return max_distance < e


def euclidean_distance(point_a, point_b):
    return np.sqrt(np.sum((point_a - point_b) ** 2))


def has_close_solutions_nd(children: list[Individ], parents, e: float,):
    n = len(children)
    max_distance = -1
    for i in range(n):
        for j in range(i+1, n):
            distance = euclidean_distance(
                children[i].solutions, children[j].solutions)
            if distance > max_distance:
                max_distance = distance

    return max_distance < e
