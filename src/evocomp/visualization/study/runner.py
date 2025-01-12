import time
from typing import Callable

from evocomp.core.objective import Objective
from evocomp.core.optimizer import Optimizer
from evocomp.visualization.study.result import StudyResult


def study(
    param_values: list[int | float],
    objective: Objective,
    setup: Callable[[int | float], Optimizer],
) -> list[StudyResult]:
    results: list[StudyResult] = []
    for value in param_values:
        algorithm = setup(value)
        start = time.time()
        algorithm.optimize(objective)
        results.append(
            StudyResult(
                fitness=algorithm.best_candidate.fitness,
                solution=algorithm.best_candidate.solution,
                history=algorithm.history,
                param_value=value,
                time=time.time() - start,
                epochs=algorithm.epochs,
            )
        )
    return results
