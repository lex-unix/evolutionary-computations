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

alogorithms = {
    'evo_strategy': EvoStrategy,
    'diff_evolution': DifferentialEvolution,
    'symbiotic_optimisation': SymbioticOptimisation,
    'simulated_annealing': SimulatedAnnealing,
    'fractal_structurization': FractalStructurization,
    'deformed_stars': DeformedStars,
}


def main():
    cli = read_cli()
    config = load_config()

    test_functions = FUNCTIONS_1D if cli.dimension == 1 else FUNCTIONS_2D if cli.dimension == 2 else FUNCTIONS_ND

    stats = Stats(
        algo=cli.algo,
        param=cli.parameter,
        plot=cli.plot,
        save_plot=cli.save_plot,
        export_format=cli.output,
    )

    selected_algo = config[cli.algo]
    params = selected_algo[cli.parameter]
    algo_config = selected_algo['default']

    Algo = alogorithms[cli.algo]
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
