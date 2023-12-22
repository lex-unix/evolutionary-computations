from src.algo.deformed_stars import DeformedStars
from src.algo.diff_evolution import DifferentialEvolution
from src.algo.evo_strategy import EvoStrategy
from src.algo.fractal_structurization import FractalStructurization
from src.algo.simulated_annealing import SimulatedAnnealing
from src.algo.symbiotic_optimisation import SymbioticOptimisation
from src.config import load_config
from src.config import read_cli
from src.lib.constants import FUNCTIONS_1D
from src.lib.constants import FUNCTIONS_2D
from src.lib.constants import FUNCTIONS_ND
from src.lib.stats import Stats
from src.lib.util import get_func_name

algorithms = {
    'evo_strategy': EvoStrategy,
    'diff_evolution': DifferentialEvolution,
    'symbiotic_optimisation': SymbioticOptimisation,
    'simulated_annealing': SimulatedAnnealing,
    'fractal_structurization': FractalStructurization,
    'deformed_stars': DeformedStars,
}

functions = {1: FUNCTIONS_1D, 2: FUNCTIONS_2D, 'nd': FUNCTIONS_ND}


def main():
    cli = read_cli()
    config = load_config()

    if cli.algo not in algorithms.keys():
        raise KeyError(f"Algorithm '{cli.algo}' not found in predefined algorithms.")

    selected_algo = config[cli.algo]

    available_params = list(config[cli.algo].keys())
    available_params.remove('default')
    if cli.parameter not in available_params:
        raise KeyError(f"Parameter '{cli.parameter}' not found for algorithm '{cli.algo}'. Available parameter: {available_params}")

    params = selected_algo[cli.parameter]
    algo_config = selected_algo['default']

    Algo = algorithms[cli.algo]
    test_functions = functions.get(cli.dimension, functions['nd'])

    stats = Stats(
        algo=cli.algo,
        param=cli.parameter,
        plot=cli.plot,
        save_plot=cli.save_plot,
        export_format=cli.output,
    )

    for func in test_functions:
        stats.objective = get_func_name(func)
        for param in params:
            stats.param_value = param
            algo_config[cli.parameter] = param
            algo = Algo(**algo_config, stats=stats)
            algo.solve(func)

        stats.display()


if __name__ == '__main__':
    main()
