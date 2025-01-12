from typing import Literal

import numpy as np
from numpy import random
from numpy.typing import NDArray

from evocomp.core.candidate import Candidate
from evocomp.core.halt_criteria import HaltCriteria
from evocomp.core.objective import Objective
from evocomp.core.optimizer import Optimizer


class EvoStrategy(Optimizer):
    def __init__(
        self,
        operation: Literal['min', 'max'],
        lmda: int = 100,
        mu: int = 20,
        std: float = 0.5,
        strategy: Literal['comma', 'plus'] = 'plus',
        epochs: int = 100,
        halt_criteria: HaltCriteria | None = None,
    ):
        super().__init__(epochs, operation, halt_criteria)
        self.lmda = lmda
        self.mu = mu
        self.strategy = strategy
        self.std = std

    def __evaluate_population(
        self,
        population: list[Candidate],
        objective: Objective,
    ) -> list[Candidate]:
        for candidate in population:
            candidate.fitness = objective.evaluate(candidate.solution)
        sorted_population = sorted(population, key=lambda x: x.fitness)[: self.mu]
        return sorted_population

    def __create_offspring(self, parent: Candidate, bounds: NDArray) -> Candidate:
        child_solution = parent.solution + self.std * random.randn(2)
        for i in range(len(child_solution)):
            child_solution[i] = np.clip(child_solution[i], bounds[i][0], bounds[i][1])
        return Candidate(child_solution)

    def generate_init_population(self, objective: Objective) -> list[Candidate]:
        bounds = objective.bounds
        x_bounds, y_bounds = bounds
        population = []
        for _ in range(self.mu):
            x = random.uniform(x_bounds[0], x_bounds[1])
            y = random.uniform(y_bounds[0], y_bounds[1])
            candidate = Candidate(np.asarray([x, y]))
            population.append(candidate)
        return population

    def compute_next_population(
        self,
        population: list[Candidate],
        objective: Objective,
    ) -> list[Candidate]:
        offspring = []
        for candidate in population:
            for _ in range(self.lmda):
                offspring.append(self.__create_offspring(candidate, objective.bounds))
        if self.strategy == 'comma':
            new_population = self.__evaluate_population(offspring, objective)
        elif self.strategy == 'plus':
            new_population = self.__evaluate_population(population + offspring, objective)

        return new_population
