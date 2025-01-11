from typing import Tuple

import numpy as np
from numpy import random
from numpy.typing import NDArray

from evocomp.core.candidate import Candidate
from evocomp.core.halt_criteria import HaltCriteria
from evocomp.core.objective import Objective
from evocomp.core.optimizer import Optimizer


class FractalStructurization(Optimizer):
    def __init__(
        self,
        m: int = 7,
        temperature: float = 100,
        std: float = 0.1,
        size: int = 50,
        epochs: int = 20,
        halt_criteria: HaltCriteria | None = None,
    ):
        super().__init__(epochs, halt_criteria)
        self.size = size
        self.m = m
        self.std = std
        self.t_max = temperature
        self.t = temperature
        self.L = 1
        self.iteration = 0
        self.radii = np.full(self.size, 1 / self.size)

    def __mean_fitness(self, population: list[Candidate]):
        return np.mean([candidate.fitness for candidate in population])

    def __get_random_indexes(self, size: int) -> Tuple[NDArray, NDArray]:
        i = random.choice(np.arange(size), size=size, replace=False)
        j = np.zeros(size, dtype=int)
        for idx in range(size):
            choices = set(range(size)) - {i[idx]}
            j[idx] = random.choice(list(choices))
        return i, j

    def __form_pairs(self, population: list[Candidate]) -> list[Tuple[Candidate, Candidate]]:
        size = len(population)
        i, j = self.__get_random_indexes(size)
        return [(population[i[k]], population[j[k]]) for k in range(size)]

    def __mean_pairs(self, pairs: list[Tuple[Candidate, Candidate]]) -> list[Candidate]:
        return [Candidate((pair[0].solution + pair[1].solution) / 2) for pair in pairs]

    def __delta(self, solution: float | NDArray, bound_diff: float | NDArray):
        return random.uniform(solution - bound_diff, solution + bound_diff)

    def __potential_candidate(
        self,
        candidate: Candidate,
        objective: Objective,
        fitness_avg: float,
    ) -> Candidate | None:
        delta = self.__delta(candidate.solution, self.__bounds_diff(objective.bounds))
        potential_ind = Candidate(candidate.solution + delta)
        potential_ind.fitness = objective.evaluate(candidate.solution)

        if (potential_ind.fitness < fitness_avg) or (
            np.random.random_sample() < np.exp(-np.min(delta) / self.t)
        ):
            return potential_ind

        return None

    def __bounds_diff(self, bounds: NDArray):
        return (bounds[:, 0] - bounds[:, 1]) / self.size

    def __population_w(self, population: list[Candidate], objective: Objective):
        fitness_avg = self.__mean_fitness(population)
        mean_pairs = self.__mean_pairs(self.__form_pairs(population[self.size * 6 :]))
        new_population = [
            self.__potential_candidate(candidate, objective, fitness_avg)
            for candidate in mean_pairs
        ]
        return [candidate for candidate in new_population if candidate is not None]

    def __population_c(self, pairs: list[Tuple[Candidate, Candidate]], objective: Objective):
        population = []
        for pair in pairs:
            r = random.choice([-1, 1])
            dims = len(objective.bounds)
            q = random.randint(0, dims)
            a, b = pair[0].solution, pair[1].solution
            c = np.where(np.arange(dims) == q, (a + b) / 2, (r == -1) * a + (r == 1) * b)
            population.append(Candidate(c))
        return population

    def __population_v(self, population: list[Candidate], objective: Objective):
        population_v = []
        for candidate, r in zip(population, self.radii):
            population_v.append(self.__offsprings(candidate, r / (self.iteration + 1), objective))
        return population_v

    def __offsprings(self, candidate: Candidate, r: float, objective: Objective) -> Candidate:
        dimms = len(objective.bounds)
        k = random.randint(dimms)
        new_solution = np.random.uniform(candidate.solution - r, candidate.solution + r, size=dimms)
        mask = np.ones(dimms, np.bool_)
        mask[k] = 0
        a = candidate.solution[k] + random.choice([-1, 1])
        b = r**2 - np.sum(new_solution[mask] - candidate.solution[mask]) ** 2
        new_solution[k] = a * b
        return Candidate(new_solution)

    def generate_init_population(self, objective: Objective) -> list[Candidate]:
        bounds = objective.bounds
        pop = random.uniform(bounds[:, 0], bounds[:, 1], (self.size, len(bounds)))
        population = [Candidate(x) for x in pop]
        self.compute_fitness(population, objective)
        self.iteration = 0
        return population

    def compute_next_population(
        self,
        population: list[Candidate],
        objective: Objective,
    ) -> list[Candidate]:
        population_v = self.__population_v(population, objective)
        population_v = population_v + population
        population_c = self.__population_c(self.__form_pairs(population_v), objective)
        self.compute_fitness(population_v, objective)
        population_v = sorted(population_v, key=lambda x: x.fitness)
        population_w = self.__population_w(population_v, objective)
        new_population = population_v + population_c + population_w
        self.compute_fitness(new_population, objective)
        new_population = sorted(new_population, key=lambda x: x.fitness)[: self.size]
        self.iteration += 1
        return new_population
