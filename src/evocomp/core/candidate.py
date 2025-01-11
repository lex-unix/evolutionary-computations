from numpy.typing import NDArray


class Candidate:
    def __init__(self, solution: NDArray, fitness: float = 0.0):
        self.solution = solution
        self.fitness = fitness

    def __str__(self) -> str:
        return f'solution={self.solution}, fitness={self.fitness}'
