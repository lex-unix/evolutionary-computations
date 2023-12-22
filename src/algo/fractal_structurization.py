from functools import partial
from typing import List
from typing import Tuple

import numpy as np
from numpy import random
from numpy._typing import NDArray

import src.lib.population_init as init
from src.lib.common import Individ
from src.lib.common import get_random_indexes
from src.lib.functions import Function
from src.lib.stats import Stats


class FractalStructurization:
    def __init__(self, stats: Stats, m: int = 7, temperature: float = 100, std: float = 0.1, size: int = 50, epochs: int = 20):
        self.size = size
        self.m = m
        self.epochs = epochs
        self.std = std
        self.t_max = temperature
        self.t = temperature
        self.L = 1
        self.fitnesses = []
        self.stats = stats

    def solve(self, objective: Function):
        if len(objective.bounds) == 1:
            return self._solve_1d(objective)
        if len(objective.bounds) == 2:
            return self._solve_2d(objective)
        if len(objective.bounds) > 2:
            return self._solve_nd(objective)

    def get_best(self, population: List[Individ]):
        best = sorted(population, key=lambda x: x.fitness)[0]
        return best.solutions, best.fitness

    def collect_fitness(self, fitness: float):
        self.fitnesses.append(fitness)

    def set_stop_criteria(self, stop_criteria, e):
        if stop_criteria is not None:
            self.stop_criteria = partial(stop_criteria, e=e)

    def evaluate_population(self, population: List[Individ], objective: Function):
        for individ in population:
            individ.fitness = objective.evaluate(individ.solutions)

    def circle_radii(self, objective: Function) -> NDArray:
        x_bounds, y_bounds = objective.bounds
        dp = x_bounds[1] - x_bounds[0]
        dq = y_bounds[1] - y_bounds[0]
        min_distance = min(dp, dq)
        radius = min_distance / self.size
        return np.full(self.size, radius)

    def calculate_fitness_avg(self, population: List[Individ]) -> float:
        return np.mean([individ.fitness for individ in population])

    def form_pairs(self, population: List[Individ]) -> List[Tuple[Individ, Individ]]:
        size = len(population)
        i, j = get_random_indexes(size)
        return [(population[i[k]], population[j[k]]) for k in range(size)]

    def pairs_average(self, pairs: List[Tuple[Individ, Individ]]) -> List[Individ]:
        return [Individ((pair[0].solutions + pair[1].solutions) / 2) for pair in pairs]

    def get_delta(self, solution: float, bound_diff: float) -> NDArray:
        return random.uniform(solution - bound_diff, solution + bound_diff)

    def create_potential_individ(self, individ: Individ, bounds_diff: NDArray, objective: Function, fitness_avg: float) -> Individ | None:
        delta = self.get_delta(individ.solutions, bounds_diff)
        potential_ind = Individ(individ.solutions + delta)
        potential_ind.fitness = objective.evaluate(individ.solutions)

        if (potential_ind.fitness < fitness_avg) or (np.random.random_sample() < np.exp(-np.min(delta) / self.t)):
            return potential_ind

        return None

    def bounds_diff(self, bounds: NDArray):
        bounds_diff = (bounds[:, 0] - bounds[:, 1]) / self.size
        return bounds_diff

    def population_w(self, population: List[Individ], objective: Function, fitness_avg: float):
        bounds_diff = self.bounds_diff(objective.bounds)
        new_population = [self.create_potential_individ(individ, bounds_diff, objective, fitness_avg) for individ in population]
        return [individ for individ in new_population if individ is not None]

    def population_c(self, pairs: List[Tuple[Individ, Individ]], objective: Function):
        population = []
        for pair in pairs:
            r = random.choice([-1, 1])
            dims = len(objective.bounds)
            q = random.randint(0, dims)
            a, b = pair[0].solutions, pair[1].solutions
            c = np.where(np.arange(dims) == q, (a + b) / 2, (r == -1) * a + (r == 1) * b)  # nopep8
            population.append(Individ(c))
        return population

    def population_v(self, population: List[Individ], radii: NDArray, objective: Function, epoch: int):
        population_v = []
        for individ, r in zip(population, radii):
            if len(objective.bounds) == 2:
                population_v.append(self.generate_offspring_2d(individ, r))
            else:
                population_v.append(self.generate_offspring_nd(individ, r / (epoch + 1), objective))  # nopep8
        return population_v

    def generate_offspring_2d(self, individ: Individ, r: float) -> Individ:
        alpha = random.choice([-1, 1])
        xh1 = random.uniform(individ.solutions[0] - 3 * self.L * r, individ.solutions[0] + 3 * self.L * r)
        xh2 = individ.solutions[1] + alpha * np.sqrt(abs(r**2 - (xh1 - individ.solutions[0]) ** 2))  # nopep8
        return Individ(np.array([xh1, xh2]))

    def generate_offspring_nd(self, individ: Individ, r: float, objective: Function) -> Individ:
        dimms = len(objective.bounds)
        k = random.randint(dimms)
        new_solution = np.random.uniform(individ.solutions - r, individ.solutions + r, size=dimms)  # nopep8
        mask = np.ones(dimms, np.bool_)
        mask[k] = 0
        a = individ.solutions[k] + random.choice([-1, 1])
        b = r**2 - np.sum(new_solution[mask] - individ.solutions[mask]) ** 2  # nopep8
        new_solution[k] = a * b
        return Individ(new_solution)

    def _solve_nd(self, objective: Function):
        population = init.uniform_population(self.size, objective.bounds)
        radii = np.full(self.size, 1 / self.size)
        self.evaluate_population(population, objective)
        _, best_fitness = self.get_best(population)
        self.collect_fitness(best_fitness)
        for epoch in range(self.epochs):
            population_v = self.population_v(population, radii, objective, epoch)  # nopep8
            population_v = population_v + population
            population_c = self.population_c(self.form_pairs(population_v), objective)  # nopep8

            self.evaluate_population(population_v, objective)
            population_v = sorted(population_v, key=lambda x: x.fitness)
            fitness_avg = self.calculate_fitness_avg(population_v)

            worst_pairs = self.form_pairs(population_v[self.size * 6 :])
            population_w = self.population_w(self.pairs_average(worst_pairs), objective, fitness_avg)  # nopep8

            new_population = population_v + population_c + population_w
            self.evaluate_population(new_population, objective)
            new_population = sorted(new_population, key=lambda x: x.fitness)[: self.size]  # nopep8

            _, best_fitness = self.get_best(new_population)
            self.collect_fitness(best_fitness)

            if hasattr(self, 'stop_criteria'):
                if self.stop_criteria(parents=population, children=new_population):
                    print(epoch + 1)
                    return self.get_best(new_population)

            population = new_population

        best, best_fitness = self.get_best(population)
        self.stats.record_solution(x=best, f=best_fitness, fitness_evolution=self.fitnesses)
        return best, best_fitness

    def _solve_2d(self, objective: Function):
        population = init.uniform_population(self.size, objective.bounds)
        self.evaluate_population(population, objective)
        radii = self.circle_radii(objective)
        _, best_fitness = self.get_best(population)
        self.collect_fitness(best_fitness)
        for epoch in range(self.epochs):
            population_v = self.population_v(population, radii, objective, epoch)  # nopep8
            population_v = population_v + population

            self.evaluate_population(population_v, objective)
            population_v = sorted(population_v, key=lambda x: x.fitness)
            fitness_avg = self.calculate_fitness_avg(population_v[: self.size])

            best_pairs = self.form_pairs(population_v[: 2 * self.size])
            worst_pairs = self.form_pairs(population_v[6 * self.size :])

            population_best_2n = self.pairs_average(best_pairs)
            population_w = self.population_w(self.pairs_average(worst_pairs), objective, fitness_avg)  # nopep8
            new_population = population_v + population_best_2n + population_w
            self.evaluate_population(new_population, objective)
            new_population = sorted(new_population, key=lambda x: x.fitness)[: self.size]  # nopep8

            _, best_fitness = self.get_best(new_population)
            self.collect_fitness(best_fitness)

            if hasattr(self, 'stop_criteria'):
                if self.stop_criteria(parents=population, children=new_population):
                    print(epoch + 1)
                    return self.get_best(new_population)

            population = new_population

            self.t /= 2

        best, best_fitness = self.get_best(population)
        self.stats.record_solution(x=best, f=best_fitness, fitness_evolution=self.fitnesses)
        return best, best_fitness

    def _solve_1d(self, objective: Function):
        population = init.uniform_population(self.size, objective.bounds)
        self.evaluate_population(population, objective)
        _, best_fitness = self.get_best(population)
        self.collect_fitness(best_fitness)
        for epoch in range(self.epochs):
            population_test = []
            for individ in population:
                sl, sr, ml, mr = 0, 0, 0, 0
                for _ in range(self.m):
                    new_solution = individ.solutions + random.rand() * self.std
                    if new_solution < individ.solutions:
                        sr += new_solution
                        mr += 1
                    else:
                        sl += new_solution
                        ml += 1

                if ml == 0:
                    xr = Individ(sr / mr)
                    xr.fitness = objective.evaluate(xr.solutions)
                    population_test.append(xr)
                    continue

                if mr == 0:
                    xl = Individ(sl / ml)
                    xl.fitness = objective.evaluate(xl.solutions)
                    population_test.append(xl)
                    continue

                xl = Individ(sl / ml)
                xr = Individ(sr / mr)

                xl.fitness = objective.evaluate(xl.solutions)
                xr.fitness = objective.evaluate(xr.solutions)
                xh = xl if xl.fitness < xr.fitness else xr
                population_test.append(xh)

            combined_population = population + population_test
            new_population = sorted(combined_population, key=lambda x: x.fitness)[: self.size]  # nopep8

            _, best_fitness = self.get_best(new_population)
            self.collect_fitness(best_fitness)

            if hasattr(self, 'stop_criteria'):
                if self.stop_criteria(parents=population, children=new_population):
                    print(epoch + 1)
                    return self.get_best(new_population)

            population = new_population

        best, best_fitness = self.get_best(population)
        self.stats.record_solution(x=best, f=best_fitness, fitness_evolution=self.fitnesses)
        return best, best_fitness
