from abc import ABC
from abc import abstractmethod
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from evocomp.core.candidate import Candidate
from evocomp.core.halt_criteria import HaltCriteria
from evocomp.core.objective import Objective


class Optimizer(ABC):
    """Abstract base class for optimization algorithms.

    This class provides a common interface and basic functionality for all
    optimization algorithms.

    Args:
        epochs: Maximum number of iterations. Ignored if halt_criteria is provided.
        operation: Direction of optimization ('min' or 'max').
            - 'min': Find minimum value of objective function
            - 'max': Find maximum value of objective function
        halt_criteria: Optional convergence criteria to stop optimization
            before reaching maximum epochs. If provided, epochs parameter
            is ignored and algorithm runs until convergence.

    Note:
        - Subclasses must implement the optimization logic
        - History of best solutions is automatically tracked
        - Can run either for fixed number of epochs or until convergence
        - Use best_candidate property to access final solution
    """

    def __init__(
        self,
        epochs: int,
        operation: Literal['min', 'max'],
        halt_criteria: HaltCriteria | None = None,
    ) -> None:
        self.__epochs = epochs if halt_criteria is None else float('inf')
        self.__halt_criteria = halt_criteria
        self.__history: list[Candidate] = []
        self.__operation = operation

    @property
    def best_candidate(self) -> Candidate:
        """Returns the best solution found during optimization."""
        return self.__history[-1]

    @property
    def epochs(self) -> int:
        """Returns the number of epochs completed during optimization."""
        return len(self.__history) - 1

    @property
    def history(self) -> list[Candidate]:
        """Returns the history of best solutions for each iteration."""
        return self.__history.copy()

    @abstractmethod
    def generate_init_population(self, objective: Objective) -> list[Candidate]:
        pass

    @abstractmethod
    def compute_next_population(
        self,
        population: list[Candidate],
        objective: Objective,
    ) -> list[Candidate]:
        pass

    def optimize(self, objective: Objective) -> Candidate:
        """Run optimization process on the given objective function.

        Args:
            objective: The objective function to optimize. Must implement
                the Objective interface with bounds and evaluate method.

        Returns:
            The best candidate solution found during optimization.

        Note:
            - Solution history is recorded and accessible via .history
            - Best solution at any point is available via .best_candidate
        """
        population = self.generate_init_population(objective)
        self.__record_solutions(population)
        epoch = 0
        while epoch < self.__epochs:
            next_population = self.compute_next_population(population, objective)
            self.__record_solutions(next_population)
            if self.__should_halt(next_population, population):
                return self._select_best(next_population)
            population = next_population
            epoch += 1
        return self._select_best(population)

    def _compute_fitness(self, population: list[Candidate], objective: Objective):
        for candidate in population:
            candidate.fitness = objective.evaluate(candidate.solution)

    def _sort_population(self, population: list[Candidate]):
        return sorted(population, key=lambda x: x.fitness, reverse=self.__operation == 'max')

    def _compare_candidates(self, one: Candidate, another: Candidate) -> Candidate:
        if self.__operation == 'min':
            return one if one.fitness < another.fitness else another
        else:
            return one if one.fitness > another.fitness else another

    def _select_best(self, population: list[Candidate]) -> Candidate:
        return self._sort_population(population)[0]

    def _clip_bounds(self, solution: NDArray, bounds: NDArray) -> NDArray:
        return np.clip(solution, bounds[:, 0], bounds[:, 1])

    def __should_halt(self, children: list[Candidate], parents: list[Candidate]) -> bool:
        return self.__halt_criteria is not None and self.__halt_criteria.halt(children, parents)

    def __record_solutions(self, population: list[Candidate]):
        self.__history.append(self._select_best(population))
