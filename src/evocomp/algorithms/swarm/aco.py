import random

import numpy as np
from numpy.typing import NDArray


class AntColonyOptimization:
    def __init__(
        self,
        n_ants: int,
        rho: float,
        alpha: float,
        beta: float,
        epochs: int = 200,
        heuristic: NDArray | None = None,
    ):
        self.n_ants = n_ants
        self.evaporation_rate = rho
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs
        self.start_node = 0
        self.p_best = -np.inf
        self.k = 0
        self.best_route = None
        self.best_dist_cost = np.inf
        self.heuristic = heuristic

    def __init_pheromone(self):
        initial_pheromone = random.random()
        self.initial_pheromone = initial_pheromone
        self.pheromone = initial_pheromone * np.ones((self.n_nodes, self.n_nodes))

    def __init_route(self):
        self.route = np.ones((self.n_ants, self.n_nodes)).astype(int)

    def __init_heuristic(self, heuristic: NDArray | None = None):
        if self.heuristic is not None:
            return
        heuristic = 1 / self.distance
        heuristic[heuristic == np.inf] = 0
        self.heuristic = heuristic

    def __choose_next_node(self, location: int, unvisited: NDArray):
        assert self.heuristic is not None
        pheromone_feat = np.power(self.pheromone[location, unvisited], self.alpha)
        visibility_feat = np.power(self.heuristic[location, unvisited], self.beta)
        features = np.multiply(pheromone_feat, visibility_feat)
        feat_sum = np.sum(features)

        probabilities = features / feat_sum

        cummulative = np.cumsum(probabilities)
        chosen_index = np.flatnonzero(cummulative > random.random())[0]

        next_node = np.flatnonzero(unvisited)[chosen_index]

        if np.max(probabilities) > self.p_best:
            self.p_best = np.max(probabilities)
            self.k = len(probabilities)

        return next_node

    def __get_min_route(self):
        dist_cost = np.sum(self.distance[self.route[:, :-1], self.route[:, 1:]], axis=1)
        dist_min_index = np.argmin(dist_cost)
        dist_min_cost = dist_cost[dist_min_index]
        min_route = self.route[dist_min_index, :]
        return min_route, dist_min_cost

    def __update_best_ant_pheromone(self, min_route: NDArray, dist_min_cost: int):
        self.pheromone = (1 - self.evaporation_rate) * self.pheromone
        for j in range(self.n_nodes - 1):
            current_city = min_route[j]
            next_city = min_route[j + 1]
            self.pheromone[current_city, next_city] += 1 / dist_min_cost

    def __compute_tau_max(self, dist_min_cost: float):
        return (1 / self.evaporation_rate) * (1 / dist_min_cost)

    def __compute_tau_min(self, tau_max: float, probability_dec: float):
        return (tau_max * (1 - probability_dec)) / (self.k * probability_dec)

    def __clip_pheromone(self, tau_min: float, tau_max: float):
        self.pheromone = np.clip(self.pheromone, tau_min, tau_max)

    def solve(self, distance: NDArray, heuristic: NDArray | None = None):
        self.distance = distance
        self.n_nodes = len(distance)
        self.__init_route()
        self.__init_pheromone()
        self.__init_heuristic(heuristic)

        for epoch in range(self.epochs):
            for i in range(self.n_ants):
                self.route[i, 0] = self.start_node
                unvisited = np.ones(self.n_nodes, np.bool_)
                for j in range(self.n_nodes - 1):
                    location = self.route[i, j]
                    unvisited[location] = False
                    next_node = self.__choose_next_node(location, unvisited)
                    self.route[i, j + 1] = next_node

                # self.route[i, self.n_nodes] = self.start_node

            min_route, dist_min_cost = self.__get_min_route()

            if dist_min_cost < self.best_dist_cost:
                self.best_dist_cost = dist_min_cost
                self.best_route = min_route

            probability_dec = np.power(self.p_best, 1 / (self.n_nodes - 1))
            self.__update_best_ant_pheromone(min_route, dist_min_cost)

            tau_max = self.__compute_tau_max(dist_min_cost)
            tau_min = self.__compute_tau_min(tau_max, probability_dec)
            self.__clip_pheromone(tau_min, tau_max)

        return self.best_route, self.best_dist_cost
