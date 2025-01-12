from typing import Literal

import numpy as np
from numpy import random

from evocomp.core.candidate import Candidate
from evocomp.core.halt_criteria import HaltCriteria
from evocomp.core.objective import Objective
from evocomp.core.optimizer import Optimizer


class SimulatedAnnealing(Optimizer):
    """Simulated Annealing (SA) algorithm for global optimization.

    SA is inspired by the annealing process in metallurgy. It uses temperature-based
    probabilistic acceptance of worse solutions to escape local optima. The algorithm
    gradually reduces temperature, becoming more selective in accepting solutions.

    Args:
        operation: Direction of optimization ('min' or 'max').
        temperature: Initial temperature. Higher values increase the probability
            of accepting worse solutions early in the search.
        std: Standard deviation for Gaussian perturbation. Controls the step
            size when generating new solutions.
        epochs: Maximum number of iterations.
        halt_criteria: Optional convergence criteria to stop optimization
            before reaching maximum epochs.

    Note:
        - Uses Gaussian (normal) distribution for generating new solutions
        - Temperature decreases linearly with iterations (T = T0/iteration)
        - Maintains only one solution at a time
    """

    def __init__(
        self,
        operation: Literal['min', 'max'],
        temperature: float,
        std: float,
        epochs: int = 500,
        halt_criteria: HaltCriteria | None = None,
    ):
        super().__init__(epochs, operation, halt_criteria)
        self.temperature = temperature
        self.std = std
        self.iteration = 0

    def generate_init_population(self, objective: Objective) -> list[Candidate]:
        initial_solution = np.array(
            [random.uniform(bound[0], bound[1]) for bound in objective.bounds]
        )
        candidate = Candidate(initial_solution, objective.evaluate(initial_solution))
        self.iteration = 0
        return [candidate]

    def compute_next_population(
        self,
        population: list[Candidate],
        objective: Objective,
    ) -> list[Candidate]:
        current_candidate = population[0]
        new_state = current_candidate.solution + random.randn(len(objective.bounds)) * self.std
        new_candidate = Candidate(new_state, objective.evaluate(new_state))
        t = self.temperature / self.iteration
        diff = new_candidate.fitness - current_candidate.fitness
        if diff < 0 or random.randn() < np.exp(-diff / t):
            current_candidate = new_candidate
        self.iteration += 1
        return [current_candidate]
