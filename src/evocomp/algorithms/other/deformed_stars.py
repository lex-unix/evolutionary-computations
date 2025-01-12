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
    def __init__(
        self,
        operation: Literal['min', 'max'],
        k: int = 3,
        compression_rate: float = 4.0,
        std: float = 0.1,
        a: int = 3,
        size: int = 10,
        epochs: int = 50,
        halt_criteria: HaltCriteria | None = None,
    ):
        super().__init__(epochs, operation, halt_criteria)
        self.size = size
        self.compression_rate = compression_rate
        self.std = std
        self.k = k
        self.a = a
        self.alpha = np.radians(90)
        self.beta = np.radians(140)

    def __get_best(self, population: list[Candidate]):
        best = sorted(population, key=lambda x: x.fitness)[0]
        return best.solution, best.fitness

    def __get_random_indexes(self):
        i = random.choice(np.arange(self.size), size=self.size, replace=False)
        j = np.zeros(self.size, dtype=int)

        for idx in range(self.size):
            choices = set(range(self.size)) - {i[idx]}
            j[idx] = np.random.choice(list(choices))

        return i, j

    def __correct_bounds(self, candidate: Candidate, objective: Objective):
        x_bounds, y_bounds = objective.bounds
        x, y = candidate.solution

        if x < x_bounds[0] or x > x_bounds[1]:
            x = random.uniform(x_bounds[0], x_bounds[1])
        if y < y_bounds[0] or y > y_bounds[1]:
            y = random.uniform(y_bounds[0], y_bounds[1])

        candidate.solution = np.array([x, y])

    def __correct_bounds_nd(self, candidate: Candidate, objective: Objective):
        for i, bound in enumerate(objective.bounds):
            if candidate.solution[i] < bound[0] or candidate.solution[i] > bound[1]:
                candidate.solution[i] = random.uniform(bound[0], bound[1])
        return candidate

    def __parallel_transfer(self, candidate: Candidate):
        x, y = candidate.solution
        x_new = x + self.a * np.cos(self.alpha)
        y_new = y + self.a * np.sin(self.alpha)
        return Candidate(np.array([x_new, y_new]))

    def __rotate(self, candidate_a: Candidate, candidate_b: Candidate):
        x_a, y_a = candidate_a.solution
        x_b, y_b = candidate_b.solution
        if candidate_a.fitness < candidate_b.fitness:
            x_new = x_b + (x_b - x_a) * np.cos(self.beta) - (y_b - y_a) * np.sin(self.beta)
            y_new = y_b + (x_b - x_a) * np.sin(self.beta) + (y_b - y_a) * np.cos(self.beta)
            new_candidate = Candidate(np.array([x_new, y_new]))
            return candidate_a, new_candidate
        else:
            x_new = x_a + (x_a - x_b) * np.cos(self.beta) - (y_a - y_b) * np.sin(self.beta)
            y_new = y_a + (x_a - x_b) * np.sin(self.beta) + (y_a - y_b) * np.cos(self.beta)
            new_candidate = Candidate(np.array([x_new, y_new]))
            return candidate_b, new_candidate

    def __compress(self, candidate_a: Candidate, candidate_b: Candidate):
        x_a, y_a = candidate_a.solution
        x_b, y_b = candidate_b.solution

        x_new = (x_a + x_b) / self.compression_rate
        y_new = (y_a + y_b) / self.compression_rate
        new_candidate = Candidate(np.array([x_new, y_new]))

        if candidate_a.fitness < candidate_b.fitness:
            return candidate_a, new_candidate
        else:
            return candidate_b, new_candidate

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

    def __find_min_fitness_points(self, triangles: list[list[Candidate]]) -> list[Candidate]:
        min_fitness_points = []
        for triangle in triangles:
            min_point = min(triangle, key=lambda candidate: candidate.fitness)
            min_fitness_points.append(min_point)
        return min_fitness_points

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
            k, L = rotation_axis
            temp_k = point.solution[k]
            point.solution[k] = point.solution[k] * np.cos(self.alpha) - point.solution[L] * np.sin(
                self.alpha
            )
            point.solution[L] = temp_k * np.sin(self.alpha) + point.solution[L] * np.cos(self.alpha)
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
        self.compute_fitness(population, objective)
        return population

    def compute_next_population(
        self, population: list[Candidate], objective: Objective
    ) -> list[Candidate]:
        triangles = self.__form_triangles(population)
        centroids = self.__find_centroids(triangles)
        min_fitness_points = self.__find_min_fitness_points(triangles)

        triangles_r = [
            self.__get_r_triangles(t, c, m)
            for (t, c, m) in zip(triangles, centroids, min_fitness_points)
        ]
        triangles_q = [self.__get_q_triangle(t, objective) for t in triangles]
        triangles_u = [self.__get_u_triangle(t, m) for t, m in zip(triangles, min_fitness_points)]

        combined_population = [
            self.__correct_bounds_nd(candidate, objective)
            for triangle in (triangles_r + triangles_q + triangles_u)
            for candidate in triangle
        ]
        self.compute_fitness(combined_population, objective)
        new_population = sorted(combined_population, key=lambda x: x.fitness)[: self.size]

        return new_population
