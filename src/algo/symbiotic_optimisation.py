import copy
from functools import partial
from typing import Callable
from typing import List

import numpy as np
from numpy import random
from numpy._typing import NDArray

import src.lib.population_init as init
from src.lib.common import Individ
from src.lib.functions import Function
from src.lib.stats import Stats


class SymbioticOptimisation:
    def __init__(self, bf1: int, bf2: int, stats: Stats, epochs: int = 100, size: int = 100):
        self.bf1 = bf1
        self.bf2 = bf2
        self.epochs = epochs
        self.size = size
        self.fitnesses = []
        self.stats = stats

    def set_stop_criteria(self, stop_criteria: Callable[List[Individ], List[Individ]], e: float):
        if stop_criteria is not None:
            self.stop_criteria = partial(stop_criteria, e=e)

    def evaluate_population(self, population: List[Individ], objective: Function):
        for individ in population:
            individ.fitness = objective.evaluate(individ.solutions)
        return sorted(population, key=lambda x: x.fitness)

    def evaluate(self, individ: Individ, objective: Function):
        individ.fitness = objective.evaluate(individ.solutions)

    def get_random_index(self, current_index: int):
        idx_pool = np.delete(np.arange(self.size), current_index)
        return random.choice(idx_pool)

    def create_parasite(self, individ: Individ, bounds: NDArray) -> Individ:
        parasite = copy.deepcopy(individ)
        random_index = random.randint(len(parasite.solutions))
        parasite.solutions[random_index] = random.uniform(-1, 1)
        return parasite

    def collect_best_fitness(self, fitness: float):
        self.fitnesses.append(fitness)

    def solve(self, objective: Function):
        population = init.uniform_population(self.size, objective.bounds)
        for epoch in range(self.epochs):
            population = self.evaluate_population(population, objective)
            best, best_solution = population[0].fitness, population[0].solutions
            self.collect_best_fitness(best)
            new_population = population[:]
            for i in range(len(population)):
                j = self.get_random_index(i)

                mutual = (population[i].solutions + population[j].solutions) / 2  # nopep8
                i_new = population[i].solutions + random.rand() * (best_solution - mutual * self.bf1)  # nopep8
                j_new = population[j].solutions + random.rand() * (best_solution - mutual * self.bf2)  # nopep8

                i_new = Individ(i_new)
                j_new = Individ(j_new)
                self.evaluate(i_new, objective)
                self.evaluate(j_new, objective)

                if i_new.fitness < population[i].fitness:
                    new_population[i] = i_new
                if j_new.fitness < population[j].fitness:
                    new_population[j] = j_new

                j = self.get_random_index(i)

                i_new = population[i].solutions + random.uniform(-1, 1) * (best_solution - population[j].solutions)  # nopep8
                i_new = Individ(i_new)
                self.evaluate(i_new, objective)

                if i_new.fitness < population[i].fitness:
                    new_population[i] = i_new

                parasite = self.create_parasite(population[i], objective.bounds)  # nopep8
                self.evaluate(parasite, objective)

                if parasite.fitness < population[i].fitness:
                    new_population[i] = parasite

            if hasattr(self, 'stop_criteria'):
                if self.stop_criteria(parents=population, children=new_population):
                    individ = sorted(new_population, key=lambda x: x.fitness)[0]  # nopep8
                    best, best_solution = individ.fitness, individ.solutions
                    self.collect_best_fitness(best)
                    return best, best_solution, epoch + 1

            population = new_population

        population = self.evaluate_population(population, objective)
        best, best_solution = population[0].fitness, population[0].solutions
        self.collect_best_fitness(best)
        self.stats.record_solution(x=best_solution, f=best, fitness_evolution=self.fitnesses)
        return best, best_solution
