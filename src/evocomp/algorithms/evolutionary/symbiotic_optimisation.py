import copy
from typing import Literal

import numpy as np
from numpy import random
from numpy.typing import NDArray

from evocomp.core.candidate import Candidate
from evocomp.core.halt_criteria import HaltCriteria
from evocomp.core.objective import Objective
from evocomp.core.optimizer import Optimizer


class SymbioticOptimisation(Optimizer):
    def __init__(
        self,
        operation: Literal['min', 'max'],
        bf1: int,
        bf2: int,
        epochs: int = 100,
        size: int = 100,
        halt_criteria: HaltCriteria | None = None,
    ):
        super().__init__(epochs, operation, halt_criteria)
        self.bf1 = bf1
        self.bf2 = bf2
        self.size = size

    def __evaluate(self, candidate: Candidate, objective: Objective):
        candidate.fitness = objective.evaluate(candidate.solution)

    def __get_random_index(self, current_index: int):
        idx_pool = np.delete(np.arange(self.size), current_index)
        return random.choice(idx_pool)

    def __create_parasite(self, candidate: Candidate, bounds: NDArray) -> Candidate:
        parasite = copy.deepcopy(candidate)
        random_index = random.randint(len(parasite.solution))
        parasite.solution[random_index] = random.uniform(-1, 1)
        return parasite

    def generate_init_population(self, objective: Objective) -> list[Candidate]:
        bounds = objective.bounds
        pop = random.uniform(bounds[:, 0], bounds[:, 1], (self.size, len(bounds)))
        population = [Candidate(x) for x in pop]
        for candidate in population:
            self.__evaluate(candidate, objective)
        return population

    def compute_next_population(
        self,
        population: list[Candidate],
        objective: Objective,
    ) -> list[Candidate]:
        population = sorted(population, key=lambda x: x.fitness)
        new_population = population[:]
        best_solution = population[0].solution
        for i in range(len(population)):
            j = self.__get_random_index(i)

            mutual = (population[i].solution + population[j].solution) / 2
            i_new = population[i].solution + random.rand() * (best_solution - mutual * self.bf1)
            j_new = population[j].solution + random.rand() * (best_solution - mutual * self.bf2)

            i_new = Candidate(i_new)
            j_new = Candidate(j_new)
            self.__evaluate(i_new, objective)
            self.__evaluate(j_new, objective)

            if i_new.fitness < population[i].fitness:
                new_population[i] = i_new
            if j_new.fitness < population[j].fitness:
                new_population[j] = j_new

            j = self.__get_random_index(i)

            i_new = population[i].solution + random.uniform(-1, 1) * (
                best_solution - population[j].solution
            )
            i_new = Candidate(i_new)
            self.__evaluate(i_new, objective)

            if i_new.fitness < population[i].fitness:
                new_population[i] = i_new

            parasite = self.__create_parasite(population[i], objective.bounds)
            self.__evaluate(parasite, objective)

            if parasite.fitness < population[i].fitness:
                new_population[i] = parasite

        return new_population
