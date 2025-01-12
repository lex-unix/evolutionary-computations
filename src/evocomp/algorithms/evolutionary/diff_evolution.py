from typing import Literal

import numpy as np
from numpy import random
from numpy.typing import NDArray

from evocomp.core.candidate import Candidate
from evocomp.core.halt_criteria import HaltCriteria
from evocomp.core.objective import Objective
from evocomp.core.optimizer import Optimizer


class DifferentialEvolution(Optimizer):
    def __init__(
        self,
        operation: Literal['min', 'max'],
        f: float,
        epochs: int = 100,
        size: int = 100,
        crossover_rate: float = 0.3,
        halt_criteria: HaltCriteria | None = None,
    ):
        super().__init__(epochs, operation, halt_criteria)
        self.size = size
        self.crossover_rate = crossover_rate
        self.f = f

    def __create_mutant(self, population: list[Candidate], idxs):
        i, j, k = idxs
        mutant = population[k].solution + self.f * (population[i].solution - population[j].solution)
        return Candidate(mutant)

    def __create_trial(self, initial: Candidate, mutant: Candidate) -> Candidate:
        trial = []
        for i in range(len(mutant.solution)):
            random_index = random.randint(len(mutant.solution))
            if random.rand() <= self.crossover_rate or i == random_index:
                trial.append(mutant.solution[i])
            else:
                trial.append(initial.solution[i])
        return Candidate(np.asarray(trial))

    def __correct_bounds(self, candidate: Candidate, bounds: NDArray) -> Candidate:
        for i in range(len(bounds)):
            min_bound, max_bound = bounds[i]
            var = candidate.solution[i]
            if var > max_bound:
                var = max_bound
            elif var < min_bound:
                var = min_bound
            candidate.solution[i] = var
        return candidate

    def generate_init_population(self, objective: Objective) -> list[Candidate]:
        bounds = objective.bounds
        dims = len(bounds)
        population = [
            Candidate(bounds[:, 0] + random.rand(dims) * (bounds[:, 1] - bounds[:, 0]))
            for _ in range(self.size)
        ]
        return population

    def compute_next_population(
        self, population: list[Candidate], objective: Objective
    ) -> list[Candidate]:
        new_population: list[Candidate] = []
        for i, candidate in enumerate(population):
            idxs_pool = np.delete(np.arange(self.size), i)
            idxs = np.random.choice(idxs_pool, 3, replace=False)

            mutant = self.__create_mutant(population, idxs)
            mutant = self.__correct_bounds(mutant, objective.bounds)

            trial = self.__create_trial(candidate, mutant)

            candidate.fitness = objective.evaluate(candidate.solution)
            trial.fitness = objective.evaluate(trial.solution)

            if candidate.fitness < trial.fitness:
                new_population.append(candidate)
            else:
                new_population.append(trial)

        return new_population
