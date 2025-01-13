from typing import Literal

import numpy as np
from numpy import random

from evocomp.core.candidate import Candidate
from evocomp.core.halt_criteria import HaltCriteria
from evocomp.core.objective import Objective
from evocomp.core.optimizer import Optimizer


class BeeColony(Optimizer):
    """Artificial Bee Colony (ABC) algorithm for global optimization.

    ABC simulates the foraging behavior of honey bees. The colony consists of three types
    of bees: employed bees, onlooker bees, and scout bees. Each employed bee is associated
    with a food source and shares information with onlooker bees. Scout bees search for
    new food sources when current ones are exhausted.

    Args:
        operation: Direction of optimization ('min' or 'max').
        max_stagnation: Maximum number of iterations without improvement before
            abandoning a food source.
        epochs: Maximum number of iterations.
        size: Colony size (number of employed bees). Total population will be
            2 × size (employed + onlooker bees).
        halt_criteria: Optional convergence criteria to stop optimization
            before reaching maximum epochs.

    Note:
        - Lower max_stagnation values increase exploration
    """

    def __init__(
        self,
        operation: Literal['min', 'max'],
        max_stagnation=10,
        epochs=100,
        size=100,
        halt_criteria: HaltCriteria | None = None,
    ):
        super().__init__(epochs, operation, halt_criteria)
        self.__size = size
        self.__max_stagnation = max_stagnation
        self.__stagnation_count = [0] * size

    def __fiti(self, bee: Candidate) -> float:
        if bee.fitness >= 0:
            return 1 / (1 + bee.fitness)
        return 1 + np.abs(bee.fitness)

    def __food_source(self, population: list[Candidate], current_index: int) -> np.ndarray:
        current_solution = population[current_index].solution
        idx_pool = np.delete(np.arange(self.__size), current_index)
        random_bee: Candidate = population[random.choice(idx_pool)]

        phi = random.uniform(-1, 1, size=len(current_solution))
        new_solution = current_solution + phi * (current_solution - random_bee.solution)

        return new_solution

    def __employed_bee_phase(self, population: list[Candidate], objective: Objective):
        for i in range(self.__size):
            bee = population[i]
            new_solution = self.__food_source(population, i)
            new_solution = self._clip_bounds(new_solution, objective.bounds)
            fitness = objective.evaluate(new_solution)
            new_bee = Candidate(new_solution, fitness)

            if self.__fiti(new_bee) > self.__fiti(bee):
                population[i] = new_bee

    def __onlooker_bee_phase(self, population: list[Candidate], objective: Objective):
        fitis = np.array([self.__fiti(bee) for bee in population])
        probabilities = fitis / np.sum(fitis)

        for i in range(self.__size):
            if random.random() < probabilities[i]:
                new_solution = self.__food_source(population, i)
                new_solution = self._clip_bounds(new_solution, objective.bounds)
                fitness = objective.evaluate(new_solution)
                new_bee = Candidate(new_solution, fitness)

                if self.__fiti(new_bee) > self.__fiti(population[i]):
                    population[i] = new_bee

    def __scout_bee_phase(self, population: list[Candidate], objective: Objective):
        best_bee = min(population, key=lambda x: x.fitness)
        for i in range(self.__size):
            if population[i].fitness == best_bee.fitness:
                if self.__stagnation_count[i] >= self.__max_stagnation:
                    new_solution = random.uniform(
                        objective.bounds[:, 0],
                        objective.bounds[:, 1],
                        size=len(objective.bounds),
                    )
                    fitness = objective.evaluate(new_solution)
                    new_bee = Candidate(new_solution, fitness)
                    population[i] = new_bee
                    self.__stagnation_count[i] = 0
                else:
                    self.__stagnation_count[i] += 1

    def generate_init_population(self, objective: Objective) -> list[Candidate]:
        population = []
        for _ in range(self.__size):
            solution = random.uniform(objective.bounds[:, 0], objective.bounds[:, 1])
            fitness = objective.evaluate(solution)
            population.append(Candidate(solution, fitness))
        return population

    def compute_next_population(
        self,
        population: list[Candidate],
        objective: Objective,
    ) -> list[Candidate]:
        new_population = population[:]
        self.__employed_bee_phase(new_population, objective)
        self.__onlooker_bee_phase(new_population, objective)
        self.__scout_bee_phase(new_population, objective)
        return new_population
