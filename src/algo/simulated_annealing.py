import numpy as np
from numpy import random

from src.lib.functions import Function


class SimulatedAnnealing:
    def __init__(self, t: float, std: float, epochs: int = 500):
        self.t = t
        self.std = std
        self.epochs = epochs
        self.costs = []

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

        return curr_state, curr_cost
