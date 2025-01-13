from typing import Literal

import numpy as np
from numpy import random
from numpy.typing import NDArray

from evocomp.core.candidate import Candidate
from evocomp.core.halt_criteria import HaltCriteria
from evocomp.core.objective import Objective
from evocomp.core.optimizer import Optimizer


class DifferentialEvolution(Optimizer):
    """Differential Evolution (DE) algorithm for global optimization.

    DE is a population-based optimization algorithm that uses vector differences
    for perturbing the vector population.

    Args:
        operation: Direction of optimization ('min' or 'max').
        f: Differential weight (mutation factor) in range [0, 2].
            Controls the amplification of differential variation.
        crossover_rate: Probability of crossover in range [0, 1].
            Higher values increase exploration.
        epochs: Maximum number of iterations.
        size: Population size. Larger populations provide better exploration
            but require more function evaluations.
        halt_criteria: Optional convergence criteria to stop optimization
            before reaching maximum epochs.
    """

    def __init__(
        self,
        operation: Literal['min', 'max'],
        f: float,
        crossover_rate: float = 0.3,
        epochs: int = 100,
        size: int = 100,
        halt_criteria: HaltCriteria | None = None,
    ):
        super().__init__(epochs, operation, halt_criteria)
        self.f = f
        self.crossover_rate = crossover_rate
        self.size = size

    def __create_mutant(self, population: list[Candidate], idxs: NDArray) -> Candidate:
        i, j, k = idxs
        mutant = population[k].solution + self.f * (population[i].solution - population[j].solution)
        return Candidate(mutant)

    def __create_trial(self, initial: Candidate, mutant: Candidate) -> Candidate:
        trial = np.array([])
        for i in range(len(mutant.solution)):
            random_index = random.randint(len(mutant.solution))
            if random.rand() <= self.crossover_rate or i == random_index:
                trial = np.append(trial, mutant.solution[i])
            else:
                trial = np.append(trial, initial.solution[i])
        return Candidate(trial)

    def generate_init_population(self, objective: Objective) -> list[Candidate]:
        bounds = objective.bounds
        solutions = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.size, len(bounds)))
        return [Candidate(solution) for solution in solutions]

    def compute_next_population(
        self,
        population: list[Candidate],
        objective: Objective,
    ) -> list[Candidate]:
        new_population: list[Candidate] = []
        for i, candidate in enumerate(population):
            idxs_pool = np.delete(np.arange(self.size), i)
            idxs = np.random.choice(idxs_pool, 3, replace=False)

            mutant = self.__create_mutant(population, idxs)
            mutant = self._clip_bounds(mutant, objective.bounds)

            trial = self.__create_trial(candidate, mutant)

            candidate.fitness = objective.evaluate(candidate.solution)
            trial.fitness = objective.evaluate(trial.solution)
            new_population.append(self._compare_candidates(candidate, trial))

        return new_population
