import time
from typing import Callable
from typing import Sequence
from typing import TypeVar

from evocomp.core.objective import Objective
from evocomp.core.optimizer import Optimizer
from evocomp.visualization.study.result import StudyResult

T = TypeVar('T', int, float)


def study(
    param_values: Sequence[T],
    objective: Objective,
    setup: Callable[[T], Optimizer],
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
                param_value=str(value),
                time=time.time() - start,
                epochs=algorithm.epochs,
            )
        )
    return results
