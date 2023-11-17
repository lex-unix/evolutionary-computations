from functools import partial
from typing import List

import numpy as np
from numpy import random

import src.lib.population_init as init
from src.lib.common import Individ
from src.lib.functions import Function


def init_colony(size: int, objective: Function) -> List[Individ]:
    x_bounds, y_bounds = objective.bounds
    colony = []
    for _ in range(size):
        x = x_bounds[0] + random.rand() * (x_bounds[1] - x_bounds[0])  # nopep8
        y = y_bounds[0] + random.rand() * (y_bounds[1] - y_bounds[0])  # nopep8
        bee = Individ(np.asarray([x, y]))
        colony.append(bee)
    return colony


class BeeColony:
    def __init__(self, max_stagnation=10, size=100, epochs=100):
        self.size = size
        self.epochs = epochs
        self.max_stagnation = max_stagnation
        self.stagnation_count = [0] * size
        self.fitnesses = []

    def set_stop_criteria(self, stop_criteria, e):
        if stop_criteria is not None:
            self.stop_criteria = partial(stop_criteria, e=e)

    def collect_best_fitness(self, fitness):
        self.fitnesses.append(fitness)

    def evaluate_colony(self, colony: List[Individ], objective: Function):
        for bee in colony:
            bee.fitness = objective.evaluate(bee.solutions)

    def fiti(self, bee: Individ, objective: Function) -> float:
        bee.fitness = objective.evaluate(bee.solutions)
        if bee.fitness >= 0:
            return 1 / (1 + bee.fitness)

        return 1 + np.abs(bee.fitness)

    def get_close_food_source(self, colony: List[Individ], current_index: int):
        x_old, y_old = colony[current_index].solutions

        idx_pool = np.delete(np.arange(self.size), current_index)
        k = random.choice(idx_pool)
        random_bee = colony[k]

        x_new = x_old + random.uniform(-1, 1) * (x_old - random_bee.solutions[0])  # nopep8
        y_new = y_old + random.uniform(-1, 1) * (y_old - random_bee.solutions[1])  # nopep8

        return x_new, y_new

    def employed_bee_phase(self, colony: List[Individ], objective: Function):
        x_bounds, y_bounds = objective.bounds
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        for i in range(self.size):
            bee = colony[i]

            x_new, y_new = self.get_close_food_source(colony, i)
            x_new = max(x_min, min(x_max, x_new))
            y_new = max(y_min, min(y_max, y_new))

            new_bee = Individ(np.asarray([x_new, y_new]))
            new_bee.fitness = objective.evaluate(new_bee.solutions)

            if self.fiti(new_bee, objective) > self.fiti(bee, objective):
                colony[i] = new_bee

    def onlooker_bee_phase(self, colony: List[Individ], objective: Function):
        fitis = [self.fiti(bee, objective) for bee in colony]
        fiti_sum = sum(fitis)
        probablities = [fiti / fiti_sum for fiti in fitis]

        x_bounds, y_bounds = objective.bounds
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        for i in range(self.size):
            if random.random() < probablities[i]:
                x_new, y_new = self.get_close_food_source(colony, i)
                x_new = max(x_min, min(x_max, x_new))
                y_new = max(y_min, min(y_max, y_new))

                new_bee = Individ(np.asarray([x_new, y_new]))
                new_bee.fitness = objective.evaluate(new_bee.solutions)

                if self.fiti(new_bee, objective) > self.fiti(colony[i], objective):
                    colony[i] = new_bee

    def scout_bee_phase(self, colony: List[Individ], objective: Function):
        x_bounds, y_bounds = objective.bounds
        best_bee = sorted(colony, key=lambda x: x.fitness)[0]
        for i in range(self.size):
            if colony[i].fitness == best_bee.fitness:
                if self.stagnation_count[i] >= self.max_stagnation:
                    x = random.uniform(x_bounds[0], x_bounds[1])
                    y = random.uniform(y_bounds[0], y_bounds[1])
                    bee = Individ(np.array([x, y]))
                    bee.fitness = objective.evaluate(bee.solutions)
                    colony[i] = bee
                    self.stagnation_count[i] = 0
                else:
                    self.stagnation_count[i] += 1

    def solve(self, objective: Function):
        colony = init.distrubited_population(self.size, objective.bounds)
        self.evaluate_colony(colony, objective)
        new_colony = colony[:]
        for epoch in range(self.epochs):
            self.employed_bee_phase(new_colony, objective)
            self.onlooker_bee_phase(new_colony, objective)
            self.scout_bee_phase(new_colony, objective)

            if hasattr(self, 'stop_criteria'):
                if self.stop_criteria(new_colony, colony):
                    best_bee = sorted(new_colony, key=lambda x: x.fitness)[0]
                    best, best_solution = best_bee.fitness, best_bee.solutions
                    self.collect_best_fitness(best)
                    return best, best_solution, epoch + 1

            colony = new_colony[:]
            best = sorted(colony, key=lambda x: x.fitness)[0]
            self.collect_best_fitness(best.fitness)

        best = sorted(colony, key=lambda x: x.fitness)[0]
        return best.fitness, best.solutions
