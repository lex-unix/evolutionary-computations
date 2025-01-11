from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from evocomp.visualization.study.result import StudyResult


def export_results(
    results: list[StudyResult],
    filename: str | Path,
    format: Literal['csv', 'xlsx'] = 'csv',
    param_name: str = 'Parameter',
    round_decimal: int = 5,
) -> None:
    if not results:
        return

    data = []
    for result in results:
        solution = np.round(result.solution, round_decimal)
        solution_str = ', '.join(map(str, solution))

        data.append(
            {
                param_name: result.param_value,
                'Fitness': result.fitness,
                'Solution': solution_str,
                'Time(s)': round(result.time, 3),
            }
        )

    df = pd.DataFrame(data)

    filename = Path(filename)
    if not filename.suffix:
        filename = filename.with_suffix(f'.{format}')

    if format == 'csv':
        df.to_csv(filename, index=False)
    else:
        df.to_excel(filename, index=False)
    print(f'Results exported to {filename}')
