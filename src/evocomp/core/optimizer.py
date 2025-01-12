from abc import ABC
from abc import abstractmethod

from evocomp.core.candidate import Candidate
from evocomp.core.halt_criteria import HaltCriteria
from evocomp.core.objective import Objective


class Optimizer(ABC):
    def __init__(self, epochs: int, halt_criteria: HaltCriteria | None = None) -> None:
        self.__epochs = epochs if halt_criteria is None else float('inf')
        self.__halt_criteria = halt_criteria
        self.__history: list[Candidate] = []

    @property
    def best_candidate(self) -> Candidate:
        return self.__history[-1]

    @property
    def epochs(self) -> int:
        return len(self.__history) - 1

    @property
    def history(self) -> list[Candidate]:
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
        population = self.generate_init_population(objective)
        self.__record_solutions(population)
        epoch = 0
        while epoch < self.__epochs:
            next_population = self.compute_next_population(population, objective)
            self.__record_solutions(next_population)
            if self.__should_halt(next_population, population):
                return self.__select_best(next_population)
            population = next_population
            epoch += 1
        return self.__select_best(population)

    def compute_fitness(self, population: list[Candidate], objective: Objective):
        for candidate in population:
            candidate.fitness = objective.evaluate(candidate.solution)

    def __select_best(self, population: list[Candidate]) -> Candidate:
        population_sorted = sorted(population, key=lambda x: x.fitness)
        return population_sorted[0]

    def __should_halt(self, children: list[Candidate], parents: list[Candidate]) -> bool:
        return self.__halt_criteria is not None and self.__halt_criteria.halt(children, parents)

    def __record_solutions(self, population: list[Candidate]):
        self.__history.append(self.__select_best(population))
