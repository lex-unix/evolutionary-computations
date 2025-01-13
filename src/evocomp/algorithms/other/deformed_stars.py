import copy
from random import sample
from typing import Literal

import numpy as np
from numpy import random
from numpy.typing import NDArray

from evocomp.core.candidate import Candidate
from evocomp.core.halt_criteria import HaltCriteria
from evocomp.core.objective import Objective
from evocomp.core.optimizer import Optimizer


class DeformedStars(Optimizer):
    """Deformed Stars algorithm for global optimization.

    This algorithm uses triangular patterns (stars) for exploration, where each triangle
    undergoes three types of transformations:
    - R-triangles: Formed using centroids and minimum fitness points
    - Q-triangles: Created through rotation around random axes
    - U-triangles: Generated through compression towards minimum fitness points

    Args:
        operation: Direction of optimization ('min' or 'max').
        k: Coefficient for R-triangle formation. Controls the spread of new points
            around the best solutions.
        compression_rate: Rate of compression for U-triangles towards best points.
            Higher values lead to more aggressive local search.
        a: Amplitude for parallel transfer operations. Controls step size in solution space.
        epochs: Maximum number of iterations.
        size: Population size. Should be divisible by 3 for optimal triangle formation.
        halt_criteria: Optional convergence criteria to stop optimization before reaching maximum epochs.
    """

    def __init__(
        self,
        operation: Literal['min', 'max'],
        k: int = 3,
        compression_rate: float = 4.0,
        a: int = 3,
        epochs: int = 50,
        size: int = 10,
        halt_criteria: HaltCriteria | None = None,
    ):
        super().__init__(epochs, operation, halt_criteria)
        self.size = size
        self.compression_rate = compression_rate
        self.k = k
        self.a = a
        self.alpha = np.radians(90)
        self.beta = np.radians(140)

    def __get_random_indexes(self):
        i = random.choice(np.arange(self.size), size=self.size, replace=False)
        j = np.zeros(self.size, dtype=int)

        for idx in range(self.size):
            choices = set(range(self.size)) - {i[idx]}
            j[idx] = np.random.choice(list(choices))

        return i, j

    def __correct_bounds(self, candidate: Candidate, objective: Objective):
        for i, bound in enumerate(objective.bounds):
            if candidate.solution[i] < bound[0] or candidate.solution[i] > bound[1]:
                candidate.solution[i] = random.uniform(bound[0], bound[1])
        return candidate

    def __form_triangles(self, population: list[Candidate]) -> list[list[Candidate]]:
        triangles = []
        pool = list(range(self.size))
        while len(pool) >= 3:
            random_idxs = sample(pool, 3)
            triangle = [population[idx] for idx in random_idxs]
            triangles.append(triangle)
            pool = [idx for idx in pool if idx not in random_idxs]
        return triangles

    def __find_centroids(self, triangles: list[list[Candidate]]) -> list[NDArray]:
        centroids = []
        for triangle in triangles:
            centroid = np.mean([candidate.solution for candidate in triangle], axis=0)
            centroids.append(centroid)
        return centroids

    def __find_optimal_fitness_points(self, triangles: list[list[Candidate]]) -> list[Candidate]:
        fitness_points = []
        for triangle in triangles:
            point = self._select_best(triangle)
            fitness_points.append(point)
        return fitness_points

    def __get_r_triangles(
        self, triangle: list[Candidate], centroid: NDArray, min_point: Candidate
    ) -> list[Candidate]:
        x = [point for point in triangle if point != min_point]
        y_1 = (1 / (self.k - 1)) * (self.k * min_point.solution - centroid)
        y_2 = (1 / self.k) * ((self.k - 1) * x[0].solution + y_1)
        y_3 = (1 / self.k) * ((self.k - 1) * x[1].solution + y_1)
        return [Candidate(y_1), Candidate(y_2), Candidate(y_3)]

    def __get_q_triangle(self, triangle: list[Candidate], objective: Objective):
        triangle_new = copy.deepcopy(triangle)
        for idx, point in enumerate(triangle_new):
            rotation_axis = np.random.choice(
                np.arange(len(objective.bounds)), size=2, replace=False
            )
            k, l = rotation_axis
            temp_k = point.solution[k]
            point.solution[k] = point.solution[k] * np.cos(self.alpha) - point.solution[l] * np.sin(
                self.alpha
            )
            point.solution[l] = temp_k * np.sin(self.alpha) + point.solution[l] * np.cos(self.alpha)
        return triangle_new

    def __get_u_triangle(self, triangle: list[Candidate], min_point: Candidate):
        new_triangle = []
        for point in triangle:
            if point != min_point:
                numerator = self.compression_rate * min_point.solution + point.solution
                denominator = 1 + self.compression_rate
                new_point_solution = numerator / denominator
            else:
                new_point_solution = point.solution
            new_triangle.append(Candidate(new_point_solution))
        return new_triangle

    def generate_init_population(self, objective: Objective) -> list[Candidate]:
        population = []
        bounds = objective.bounds
        bounds_len = len(bounds)
        for _ in range(self.size):
            x = np.zeros(bounds_len)
            for i in range(bounds_len):
                x[i] = random.uniform(bounds[i, 0], bounds[i, 1])
            population.append(Candidate(x))
        self._compute_fitness(population, objective)
        return population

    def compute_next_population(
        self, population: list[Candidate], objective: Objective
    ) -> list[Candidate]:
        triangles = self.__form_triangles(population)
        centroids = self.__find_centroids(triangles)
        fitness_points = self.__find_optimal_fitness_points(triangles)

        triangles_r = [
            self.__get_r_triangles(t, c, m)
            for (t, c, m) in zip(triangles, centroids, fitness_points)
        ]
        triangles_q = [self.__get_q_triangle(t, objective) for t in triangles]
        triangles_u = [self.__get_u_triangle(t, m) for t, m in zip(triangles, fitness_points)]

        combined_population = [
            self.__correct_bounds(candidate, objective)
            for triangle in (triangles_r + triangles_q + triangles_u)
            for candidate in triangle
        ]
        self._compute_fitness(combined_population, objective)
        new_population = self._sort_population(combined_population)[: self.size]

        return new_population
