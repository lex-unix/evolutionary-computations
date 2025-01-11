import numpy as np
from numpy import random

from evocomp.core.candidate import Candidate
from evocomp.core.halt_criteria import HaltCriteria
from evocomp.core.objective import Objective
from evocomp.core.optimizer import Optimizer


class BeeColony(Optimizer):
    def __init__(
        self,
        max_stagnation=10,
        size=100,
        epochs=100,
        halt_criteria: HaltCriteria | None = None,
    ):
        super().__init__(epochs, halt_criteria)
        self.size = size
        self.max_stagnation = max_stagnation
        self.__stagnation_count = [0] * size

    def __fiti(self, bee: Candidate) -> float:
        if bee.fitness >= 0:
            return 1 / (1 + bee.fitness)
        return 1 + np.abs(bee.fitness)

    def __food_source(self, population: list[Candidate], current_index: int) -> np.ndarray:
        current_solution = population[current_index].solution
        idx_pool = np.delete(np.arange(self.size), current_index)
        random_bee: Candidate = population[random.choice(idx_pool)]

        phi = random.uniform(-1, 1, size=len(current_solution))
        new_solution = current_solution + phi * (current_solution - random_bee.solution)

        return new_solution

    def __employed_bee_phase(self, population: list[Candidate], objective: Objective):
        for i in range(self.size):
            bee = population[i]
            new_solution = self.__food_source(population, i)
            new_solution = np.clip(new_solution, objective.bounds[:, 0], objective.bounds[:, 1])
            fitness = objective.evaluate(new_solution)
            new_bee = Candidate(new_solution, fitness)

            if self.__fiti(new_bee) > self.__fiti(bee):
                population[i] = new_bee

    def __onlooker_bee_phase(self, population: list[Candidate], objective: Objective):
        fitis = np.array([self.__fiti(bee) for bee in population])
        probabilities = fitis / np.sum(fitis)

        for i in range(self.size):
            if random.random() < probabilities[i]:
                new_solution = self.__food_source(population, i)
                new_solution = np.clip(new_solution, objective.bounds[:, 0], objective.bounds[:, 1])
                fitness = objective.evaluate(new_solution)
                new_bee = Candidate(new_solution, fitness)

                if self.__fiti(new_bee) > self.__fiti(population[i]):
                    population[i] = new_bee

    def __scout_bee_phase(self, population: list[Candidate], objective: Objective):
        best_bee = min(population, key=lambda x: x.fitness)
        for i in range(self.size):
            if population[i].fitness == best_bee.fitness:
                if self.__stagnation_count[i] >= self.max_stagnation:
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
        for _ in range(self.size):
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
