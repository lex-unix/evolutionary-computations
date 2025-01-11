import matplotlib.pyplot as plt

from evocomp.visualization.study.result import StudyResult


def plot_histories(
    results: list[StudyResult],
    param_name: str = 'Parameter',
    function_name: str = '',
    algorithm_name: str = '',
    save: bool = False,
) -> None:
    plt.figure(figsize=(12, 8))

    for result in results:
        label = f'{param_name}={result.param_value}'
        plt.plot([entry.fitness for entry in result.history], label=label)

    title = 'Optimization History'
    if algorithm_name and function_name:
        title = f'{algorithm_name} on {function_name}'

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)

    if save:
        filename = f"{algorithm_name.lower().replace(' ', '_')}_{function_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    plt.show()
