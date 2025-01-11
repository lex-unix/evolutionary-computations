import numpy as np

import evocomp

# Simple distance matrix for TSP
distance_matrix = np.array(
    [
        [0, 2, 4, 1, 3],
        [2, 0, 3, 5, 2],
        [4, 3, 0, 2, 4],
        [1, 5, 2, 0, 1],
        [3, 2, 4, 1, 0],
    ]
)

optimizer = evocomp.AntColonyOptimization(
    n_ants=10,
    rho=0.1,  # evaporation rate
    alpha=1.1,  # pheromone importance
    beta=1.0,  # heuristic importance
    epochs=50,
)
route, distance = optimizer.solve(distance_matrix)
print(f'Best route: {route}')
print(f'Total distance: {distance:.2f}')
