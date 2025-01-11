from evocomp.visualization.display.config import Column
from evocomp.visualization.display.config import DisplayConfig
from evocomp.visualization.display.formatters import calculate_column_widths
from evocomp.visualization.display.formatters import print_table
from evocomp.visualization.study.result import StudyResult


def display(results: list[StudyResult], config: DisplayConfig = DisplayConfig()) -> None:
    if not results:
        print('No results to display')
        return

    if config.algorithm or config.objective:
        if config.algorithm:
            print(f'Algorithm: {config.algorithm}')
        if config.objective:
            print(f'Function:  {config.objective}')
        print()

    columns = {
        'param': Column(config.param_name, 6),
        'fitness': Column('Fitness', 10),
        'solution': Column('Solution', 12),
        'epochs': Column('Epochs', 6),
        'time': Column('Time(s)', 10),
    }

    col_widths = calculate_column_widths(results, columns, config)
    print_table(results, columns, col_widths, config)
