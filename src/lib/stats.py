from typing import Dict, List, Literal, Union

import matplotlib.pyplot as plt
import pandas as pd
from numpy._typing import NDArray

from src.lib.util import round


class Stats:
    def __init__(self, plot: bool = False, save_plot: bool = False, export_format: Literal['csv', 'excel'] = None):
        self._plot = plot
        self._save_plot = save_plot
        self._fitness_evolution: List[float] = []
        self._results: List[Dict] = []
        self._export_format = export_format
        self._algo = ''
        self._objective = ''

    @property
    def param_name(self):
        return self._param_name

    @param_name.setter
    def param_name(self, param_name: str):
        self._param_name = param_name

    @property
    def algo(self):
        return self._algo

    @algo.setter
    def algo(self, algo: str):
        self._algo = algo

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, objective: str):
        self._objective = objective

    def collect_data(self, x: NDArray, f: float, fitness_evolution: List[float], param_value: Union[int, float, str]):
        entry = {
            'param_value': param_value,
            'x': round(x),
            'f': round(f),
            'fitness_evolution': fitness_evolution
        }
        self._results.append(entry)

    def display(self):
        print(f'Optimisation by {self.algo} on {self.objective}')

        for result in self._results:
            param_name = self.param_name
            param_value = result['param_value']
            x = result['x']
            f = result['f']
            print(f'{param_name}={param_value}')
            print(f'x={x}, f()={f}')
            print()

        if self._plot:
            self._plot_graph()

        if self._export_format:
            self._export()

        self._results = []

    def _plot_graph(self):
        plt.figure(figsize=(12, 8))

        for result in self._results:
            f = result['fitness_evolution']
            label = f"{self.param_name} = {result['param_value']}"
            plt.plot(f, label=label)

        plt.title(f'Fitness evolution on {self.objective} using {self.algo}')
        plt.xlabel('Epoch')
        plt.ylabel('Fitness')
        plt.legend()

        if self._save_plot:
            plt.savefig('some shitty.png')

        plt.show()

    def _export(self):
        df = pd.DataFrame(self._results)
        df.drop('fitness_evolution', axis=1, inplace=True)
        df.to_excel('output.xlsx', index=False)
