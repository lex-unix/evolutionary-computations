from functools import partial
from typing import Callable, List

import numpy as np
from numpy import random
from numpy._typing import NDArray

import src.lib.population_init as init
from src.lib.common import Individ
from src.lib.functions import Function


class DifferentialEvolution:
    def __init__(self, f: float, epochs: int = 100, size: int = 100, crossover_rate: float = 0.3):
        self.epochs = epochs
        self.size = size
        self.crossover_rate = crossover_rate
        self.f = f
        self.fitnesses = []

    def set_stop_criteria(self, stop_criteria: Callable[List[Individ], List[Individ]], e: float):
        if stop_criteria is not None:
            self.stop_criteria = partial(stop_criteria, e=e)

    def collect_min_fitness(self, fitness: float):
        self.fitnesses.append(fitness)

    def create_mutant(self, population: List[Individ], idxs):
        i, j, k = idxs
        mutant = population[k].solutions + self.f * (population[i].solutions - population[j].solutions)  # nopep8
        return Individ(mutant)

    def create_trial(self, initial: Individ, mutant: Individ) -> Individ:
        trial = []
        for i in range(len(mutant.solutions)):
            random_index = random.randint(len(mutant.solutions))
            if random.rand() <= self.crossover_rate or i == random_index:
                trial.append(mutant.solutions[i])
            else:
                trial.append(initial.solutions[i])
        return Individ(np.asarray(trial))

    def correct_bounds(self, individ: Individ, bounds: NDArray) -> Individ:
        for i in range(len(bounds)):
            min_bound, max_bound = bounds[i]
            var = individ.solutions[i]
            if var > max_bound:
                var = max_bound
            elif var < min_bound:
                var = min_bound
            individ.solutions[i] = var
        return individ

    def solve(self, objective: Function):
        population = init.distrubited_population(self.size, objective.bounds)

        for individ in population:
            individ.fitness = objective.evaluate(individ.solutions)
        best = sorted(population, key=lambda x: x.fitness)[0]
        self.collect_min_fitness(best.fitness)

        for epoch in range(self.epochs):
            new_population = []
            for i, individ in enumerate(population):
                idxs_pool = np.delete(np.arange(self.size), i)
                idxs = np.random.choice(idxs_pool, 3, replace=False)

                mutant = self.create_mutant(population, idxs)
                mutant = self.correct_bounds(mutant, objective.bounds)

                trial = self.create_trial(individ, mutant)

                individ.fitness = objective.evaluate(individ.solutions)
                trial.fitness = objective.evaluate(trial.solutions)

                if individ.fitness < trial.fitness:
                    new_population.append(individ)
                else:
                    new_population.append(trial)

            if hasattr(self, 'stop_criteria'):
                if self.stop_criteria(parents=population, children=new_population):
                    individ = sorted(new_population, key=lambda x: x.fitness)[0]  # nopep8
                    best, best_solution = individ.fitness, individ.solutions
                    self.collect_min_fitness(best)
                    return best, best_solution, epoch + 1

            population = np.asarray(new_population)
            best = sorted(population, key=lambda x: x.fitness)[0]
            self.collect_min_fitness(best.fitness)

        best = sorted(population, key=lambda x: x.fitness)[0]
        return best.fitness, best.solutions
