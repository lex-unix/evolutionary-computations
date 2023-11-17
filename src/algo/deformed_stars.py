import copy
from functools import partial
from random import sample
from typing import List, Tuple

import numpy as np
from numpy import random
from numpy._typing import NDArray

import src.lib.population_init as init
from src.lib.common import Individ
from src.lib.functions import Function


class DeformedStars:
    def __init__(self, k: int = 3, compression_rate: float = 4.0, std: float = 0.1, a: int = 3, size: int = 10, epochs: int = 50):
        self.size = size
        self.epochs = epochs
        self.compression_rate = compression_rate
        self.alpha = np.radians(90)
        self.beta = np.radians(140)
        self.a = a
        self.std = std
        self.fitnesses = []
        self.k = k

    def init_population(self, bounds: NDArray):
        population = []
        bounds_len = len(bounds)
        for _ in range(self.size):
            x = np.zeros(bounds_len)
            for i in range(bounds_len):
                x[i] = random.uniform(bounds[i, 0], bounds[i, 1])
            population.append(Individ(x))
        return population

    def collect_fitness(self, fitness: float):
        self.fitnesses.append(fitness)

    def get_best(self, population: List[Individ]):
        best = sorted(population, key=lambda x: x.fitness)[0]
        return best.solutions, best.fitness

    def evaluate_population(self, population: List[Individ], objective: Function):
        for individ in population:
            individ.fitness = objective.evaluate(individ.solutions)

    def get_random_indexes(self):
        i = random.choice(np.arange(self.size), size=self.size, replace=False)
        j = np.zeros(self.size, dtype=int)

        for idx in range(self.size):
            choices = set(range(self.size)) - {i[idx]}
            j[idx] = np.random.choice(list(choices))

        return i, j

    def set_stop_criteria(self, stop_criteria, e):
        if stop_criteria is not None:
            self.stop_criteria = partial(stop_criteria, e=e)

    def correct_bounds(self, individ: Individ, objective: Function):
        x_bounds, y_bounds = objective.bounds
        x, y = individ.solutions

        if x < x_bounds[0] or x > x_bounds[1]:
            x = random.uniform(x_bounds[0], x_bounds[1])
        if y < y_bounds[0] or y > y_bounds[1]:
            y = random.uniform(y_bounds[0], y_bounds[1])

        individ.solutions = np.array([x, y])

    def correct_bounds_nd(self, individ: Individ, objective: Function):
        for i, bound in enumerate(objective.bounds):
            if individ.solutions[i] < bound[0] or individ.solutions[i] > bound[1]:
                individ.solutions[i] = random.uniform(bound[0], bound[1])
        return individ

    def parallel_transfer(self, individ: Individ):
        x, y = individ.solutions
        x_new = x + self.a * np.cos(self.alpha)
        y_new = y + self.a * np.sin(self.alpha)
        return Individ(np.array([x_new, y_new]))

    def rotate(self, individ_a: Individ, individ_b: Individ):
        x_a, y_a = individ_a.solutions
        x_b, y_b = individ_b.solutions
        if individ_a.fitness < individ_b.fitness:
            x_new = x_b + (x_b - x_a) * np.cos(self.beta) - (y_b - y_a) * np.sin(self.beta)  # nopep8
            y_new = y_b + (x_b - x_a) * np.sin(self.beta) + (y_b - y_a) * np.cos(self.beta)  # nopep8
            new_individ = Individ(np.array([x_new, y_new]))
            return individ_a, new_individ
        else:
            x_new = x_a + (x_a - x_b) * np.cos(self.beta) - (y_a - y_b) * np.sin(self.beta)  # nopep8
            y_new = y_a + (x_a - x_b) * np.sin(self.beta) + (y_a - y_b) * np.cos(self.beta)  # nopep8
            new_individ = Individ(np.array([x_new, y_new]))
            return individ_b, new_individ

    def compress(self, individ_a: Individ, individ_b: Individ):
        x_a, y_a = individ_a.solutions
        x_b, y_b = individ_b.solutions

        x_new = (x_a + x_b) / self.compression_rate
        y_new = (y_a + y_b) / self.compression_rate
        new_individ = Individ(np.array([x_new, y_new]))

        if individ_a.fitness < individ_b.fitness:
            return individ_a, new_individ
        else:
            return individ_b, new_individ

    def solve(self, objective: Function):
        if len(objective.bounds) == 1:
            return self._solve_1d(objective)
        if len(objective.bounds) == 2:
            return self._solve_2d(objective)
        if len(objective.bounds) > 2:
            return self._solve_nd(objective)

    def form_triangles(self, population: List[Individ]) -> List[List[Individ]]:
        triangles = []
        pool = list(range(self.size))
        while len(pool) >= 3:
            random_idxs = sample(pool, 3)
            triangle = [population[idx] for idx in random_idxs]
            triangles.append(triangle)
            pool = [idx for idx in pool if idx not in random_idxs]
        return triangles

    def find_centroids(self, triangles: Tuple[Individ]) -> List[NDArray]:
        centroids = []
        for triangle in triangles:
            centroid = np.mean([individ.solutions for individ in triangle], axis=0)  # nopep8
            centroids.append(centroid)
        return centroids

    def find_min_fitness_points(self, triangles: List[List[Individ]]) -> List[Individ]:
        min_fitness_points = []
        for triangle in triangles:
            min_point = min(triangle, key=lambda individ: individ.fitness)
            min_fitness_points.append(min_point)
        return min_fitness_points

    def get_r_triangles(self, triangle: List[Individ], centroid: NDArray, min_point: Individ) -> List[Individ]:
        x = [point for point in triangle if point != min_point]
        y_1 = (1 / (self.k-1)) * (self.k * min_point.solutions - centroid)
        y_2 = (1 / self.k) * ((self.k - 1) * x[0].solutions + y_1)
        y_3 = (1 / self.k) * ((self.k - 1) * x[1].solutions + y_1)
        return [Individ(y_1), Individ(y_2), Individ(y_3)]

    def get_q_triangle(self, triangle: List[Individ], objective: Function):
        triangle_new = copy.deepcopy(triangle)
        for idx, point in enumerate(triangle_new):
            rotation_axis = np.random.choice(np.arange(len(objective.bounds)), size=2, replace=False)  # nopep8
            k, L = rotation_axis
            temp_k = point.solutions[k]
            point.solutions[k] = point.solutions[k] * np.cos(self.alpha) - point.solutions[L] * np.sin(self.alpha)  # nopep8
            point.solutions[L] = temp_k * np.sin(self.alpha) + point.solutions[L] * np.cos(self.alpha)  # nopep8
        return triangle_new

    def get_u_triangle(self, triangle: List[Individ], min_point: Individ):
        new_triangle = []
        for point in triangle:
            if point != min_point:
                numerator = self.compression_rate * min_point.solutions + point.solutions
                denominator = 1 + self.compression_rate
                new_point_solution = numerator / denominator
            else:
                new_point_solution = point.solutions
            new_triangle.append(Individ(new_point_solution))
        return new_triangle

    def _solve_nd(self, objective: Function):
        population = init.uniform_population(self.size, objective.bounds)
        self.evaluate_population(population, objective)

        for epoch in range(self.epochs):
            triangles = self.form_triangles(population)
            centroids = self.find_centroids(triangles)
            min_fitness_points = self.find_min_fitness_points(triangles)

            triangles_r = [self.get_r_triangles(t, c, m) for (t, c, m) in zip(triangles, centroids, min_fitness_points)]  # nopep8
            triangles_q = [self.get_q_triangle(t, objective) for t in triangles]  # nopep8
            triangles_u = [self.get_u_triangle(t, m) for t, m in zip(triangles, min_fitness_points)]  # nopep8

            combined_population = [self.correct_bounds_nd(individ, objective) for triangle in (
                triangles_r + triangles_q + triangles_u) for individ in triangle]

            self.evaluate_population(combined_population, objective)
            new_population = sorted(combined_population, key=lambda x: x.fitness)[:self.size]  # nopep8

            _, best_fitness = self.get_best(new_population)
            self.collect_fitness(best_fitness)

            if hasattr(self, 'stop_criteria'):
                if self.stop_criteria(parents=population, children=new_population):
                    print(epoch + 1)
                    return self.get_best(new_population)

            population = new_population

        return self.get_best(population)

    def _solve_2d(self, objective: Function):
        population = init.uniform_population(self.size, objective.bounds)
        self.evaluate_population(population, objective)

        _, best_fitness = self.get_best(population)
        self.collect_fitness(best_fitness)

        for epoch in range(self.epochs):
            population_z = []
            i, j = self.get_random_indexes()
            for k in range(self.size):
                individ_a = population[i[k]]
                individ_b = population[j[k]]
                new_individ_a = self.parallel_transfer(individ_a)
                new_individ_b = self.parallel_transfer(individ_b)
                self.correct_bounds(new_individ_a, objective)
                self.correct_bounds(new_individ_b, objective)
                population_z.append(new_individ_a)
                population_z.append(new_individ_b)

            self.evaluate_population(population_z, objective)

            population_s = []
            i, j = self.get_random_indexes()
            for k in range(self.size):
                individ_a = population[i[k]]
                individ_b = population[j[k]]
                new_individ_a, new_individ_b = self.rotate(individ_a, individ_b)  # nopep8
                self.correct_bounds(new_individ_a, objective)
                self.correct_bounds(new_individ_b, objective)
                population_s.append(new_individ_a)
                population_s.append(new_individ_b)

            self.evaluate_population(population_s, objective)

            population_w = []
            i, j = self.get_random_indexes()
            for k in range(self.size):
                individ_a = population[i[k]]
                individ_b = population[j[k]]

                new_individ_a, new_individ_b = self.compress(individ_a, individ_b)  # nopep8
                population_w.append(new_individ_a)
                population_w.append(new_individ_b)

            self.evaluate_population(population_w, objective)

            combined_population = population_z + population_s + population_w  # nopep8
            new_population = sorted(combined_population, key=lambda x: x.fitness)[:self.size]  # nopep8

            _, best_fitness = self.get_best(new_population)
            self.collect_fitness(best_fitness)

            if hasattr(self, 'stop_criteria'):
                if self.stop_criteria(parents=population, children=new_population):
                    print(f'total epoch={epoch + 1}')
                    return self.get_best(new_population)

            population = new_population

        return self.get_best(population)

    def _solve_1d(self, objective: Function):
        bounds = objective.bounds

        population_t = init.uniform_population(self.size, objective.bounds)
        self.evaluate_population(population_t, objective)

        _, best_fitness = self.get_best(population_t)
        self.collect_fitness(best_fitness)
        for epoch in range(self.epochs):
            population_z = []

            for individ in population_t:
                new_x = individ.solutions + random.randn() * self.std
                if new_x < bounds[0, 0]:
                    new_x += (bounds[0, 0] - bounds[0, 1])
                if new_x > bounds[0, 1]:
                    new_x += (bounds[0, 1] - bounds[0, 0])
                population_z.append(Individ(new_x))

            self.evaluate_population(population_z, objective)

            population_s = []
            i, j = self.get_random_indexes()
            for k in range(self.size):
                new_x = (population_t[i[k]].solutions + population_t[j[k]].solutions) / 2  # nopep8
                population_s.append(Individ(new_x))

            self.evaluate_population(population_s, objective)
            combined_population = population_t + population_z + population_s
            new_population = sorted(combined_population, key=lambda x: x.fitness)[:self.size]  # nopep8

            _, best_fitness = self.get_best(population_t)
            self.collect_fitness(best_fitness)

            if hasattr(self, 'stop_criteria'):
                if self.stop_criteria(parents=population_t, children=new_population):
                    print(f'total epoch={epoch + 1}')
                    return self.get_best(new_population)

            population_t = new_population

        return self.get_best(population_t)
