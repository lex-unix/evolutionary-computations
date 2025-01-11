from dataclasses import dataclass

from numpy.typing import NDArray

from evocomp.core.candidate import Candidate


@dataclass
class StudyResult:
    fitness: float
    solution: NDArray
    history: list[Candidate]
    param_value: int | float
    time: float
    epochs: int
