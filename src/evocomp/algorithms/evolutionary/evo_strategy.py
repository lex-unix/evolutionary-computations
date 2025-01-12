from typing import Literal

import numpy as np
from numpy.typing import NDArray

from evocomp.core.candidate import Candidate
from evocomp.core.halt_criteria import HaltCriteria
from evocomp.core.objective import Objective
from evocomp.core.optimizer import Optimizer


class EvoStrategy(Optimizer):
    """Evolution Strategy (ES) algorithm for global optimization.

    ES is a population-based algorithm that uses mutation and selection as its main operators.
    It follows either (μ + λ) or (μ, λ) selection strategy, where μ is the number of parents
    and λ is the number of offspring.

    Args:
        operation: Direction of optimization ('min' or 'max').
        lmda: Number of offspring (λ) to generate per parent.
            Total offspring population will be μ × λ.
        mu: Number of parents (μ) to select for next generation.
            Represents population size.
        std: Standard deviation for Gaussian mutation.
            Controls mutation step size.
        strategy: Selection strategy ('plus' or 'comma').
            'plus': (μ + λ) selection, parents compete with offspring.
            'comma': (μ, λ) selection, only offspring are selected.
        epochs: Maximum number of iterations.
        halt_criteria: Optional convergence criteria to stop optimization
            before reaching maximum epochs.

    Note:
        - 'plus' strategy is more elitist and preserves best solutions
        - 'comma' strategy allows escaping local optima more easily
    """

    def __init__(
        self,
        operation: Literal['min', 'max'],
        lmda: int = 10,
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
        child_solution = parent.solution + self.std * np.random.randn(len(bounds))
        child_solution = np.clip(child_solution, bounds[:, 0], bounds[:, 1])
        return Candidate(child_solution)

    def generate_init_population(self, objective: Objective) -> list[Candidate]:
        bounds = objective.bounds
        solutions = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.mu, len(bounds)))
        return [Candidate(solution) for solution in solutions]

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
