import numpy as np

from evocomp.visualization.display.config import Column
from evocomp.visualization.display.config import DisplayConfig
from evocomp.visualization.study.result import StudyResult


def calculate_column_widths(
    results: list[StudyResult],
    columns: dict[str, Column],
    config: DisplayConfig,
) -> dict[str, int]:
    param_values_width = max(len(r.param_value) for r in results)
    param_width = max(len(config.param_name), param_values_width, columns['param'].min_width)

    first_solution = np.round(results[0].solution, config.float_precision)
    solution_str = ', '.join(str(x) for x in first_solution)
    solution_width = max(len(solution_str), columns['solution'].min_width)

    return {
        'param': param_width + columns['param'].padding,
        'fitness': columns['fitness'].min_width + columns['fitness'].padding,
        'solution': solution_width + columns['solution'].padding,
        'epochs': columns['epochs'].min_width + columns['epochs'].padding,
        'time': columns['time'].min_width + columns['time'].padding,
    }


def print_table(
    results: list[StudyResult],
    columns: dict[str, Column],
    widths: dict[str, int],
    config: DisplayConfig,
) -> None:
    print('Parameter Study Results:')

    headers = ' '.join(f'{col.name:<{widths[name]}}' for name, col in columns.items())
    print(headers)
    print('-' * sum(widths.values()))

    for result in results:
        solution_str = ', '.join(str(x) for x in np.round(result.solution, config.float_precision))
        row = (
            f"{result.param_value:<{widths['param']}} "
            f"{result.fitness:<{widths['fitness']}.{config.float_precision}f} "
            f"{solution_str:<{widths['solution']}} "
            f"{result.epochs:<{widths['epochs']}} "
            f"{result.time:<{widths['time']}.{config.time_precision}f}"
        )
        print(row)
