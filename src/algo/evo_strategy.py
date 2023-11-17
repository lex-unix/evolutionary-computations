from functools import partial
from typing import List

import numpy as np
from numpy import random
from numpy._typing import NDArray

import src.lib.population_init as init
from src.lib.common import Individ
from src.lib.functions import Function


def init_population(bounds, size) -> List[Individ]:
    x_bounds, y_bounds = bounds
    population = []
    for _ in range(size):
        x = random.uniform(x_bounds[0], x_bounds[1])
        y = random.uniform(y_bounds[0], y_bounds[1])
        individ = Individ(np.asarray([x, y]))
        population.append(individ)
    return population


class EvoStrategy:
    def __init__(self, lmda: int, mu: int, strategy: str, std: float, epochs=100):
        self.lmda = lmda
        self.mu = mu
        self.strategy = strategy
        self.epochs = epochs
        self.std = std
        self.fitnesses = []

    def evaluate_population(self, population: List[Individ], objective: Function) -> List[Individ]:
        for individ in population:
            individ.fitness = objective.evaluate(individ.solutions)
        sorted_population = sorted(
            population, key=lambda x: x.fitness)[:self.mu]
        return sorted_population

    def create_offspring(self, parent: Individ, bounds: NDArray) -> Individ:
        child_solution = parent.solutions + self.std * random.randn(2)
        for i in range(len(child_solution)):
            child_solution[i] = np.clip(child_solution[i], bounds[i][0], bounds[i][1])  # nopep8
        return Individ(child_solution)

    def collect_min_fitness(self, fitness: float):
        self.fitnesses.append(fitness)

    def set_stop_criteria(self, stop_criteria, e):
        if stop_criteria is not None:
            self.stop_criteria = partial(stop_criteria, e=e)

    def solve(self, objective: Function):
        population = init.uniform_population(self.mu, objective.bounds)
        population = self.evaluate_population(population, objective)

        best, best_solution = population[0].fitness, population[0].solutions
        self.collect_min_fitness(best)

        for epoch in range(self.epochs):
            offspring = []
            for individ in population:
                for _ in range(self.lmda // self.mu):
                    offspring.append(self.create_offspring(individ, objective.bounds))  # nopep8

            if self.strategy == "comma":
                new_population = self.evaluate_population(offspring, objective)  # nopep8
            elif self.strategy == "plus":
                new_population = self.evaluate_population(population + offspring, objective)  # nopep8

            if hasattr(self, 'stop_criteria'):
                if self.stop_criteria(parents=population, children=new_population):
                    individ = sorted(new_population, key=lambda x: individ.fitness)[0]  # nopep8
                    best, best_solution = individ.fitness, individ.solutions
                    self.collect_min_fitness(best)
                    return best, best_solution, epoch + 1

            population = new_population

            best, best_solution = population[0].fitness, population[0].solutions
            self.collect_min_fitness(best)

        return best, best_solution
