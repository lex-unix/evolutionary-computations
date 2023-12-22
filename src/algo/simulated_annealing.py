import numpy as np
from numpy import random

from src.lib.functions import Function
from src.lib.stats import Stats


class SimulatedAnnealing:
    def __init__(self, temperature: float, std: float, stats: Stats, epochs: int = 500):
        self.t = temperature
        self.std = std
        self.epochs = epochs
        self.costs = []
        self.stats = stats

    def collect_cost(self, cost: float):
        self.costs.append(cost)

    def solve(self, objective: Function):
        bounds = objective.bounds

        curr_state = np.array([])
        for b in bounds:
            x = np.random.uniform(b[0], b[1])
            curr_state = np.append(curr_state, x)

        curr_cost = objective.evaluate(curr_state)
        self.collect_cost(curr_cost)

        for b in range(self.epochs):
            new_state = curr_state + random.randn(len(bounds)) * self.std
            new_cost = objective.evaluate(new_state)

            t = self.t / (b + 1)
            diff = new_cost - curr_cost
            if diff < 0 or random.randn() < np.exp(-diff / t):
                curr_state = new_state
                curr_cost = new_cost
                self.collect_cost(curr_cost)

        self.stats.record_solution(x=curr_state, f=curr_cost, fitness_evolution=self.costs)
        return curr_state, curr_cost
