from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class DisplayConfig:
    param_name: str = 'Parameter'
    algorithm: str = ''
    objective: str = ''
    float_precision: int = 6
    time_precision: int = 2


class Column(NamedTuple):
    name: str
    min_width: int = 10
    padding: int = 4
