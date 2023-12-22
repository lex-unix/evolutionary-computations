import os
from typing import Dict
from typing import List
from typing import Literal
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
from numpy._typing import NDArray

from src.lib import util


class Stats:
    def __init__(self, algo: str, param: str, plot: bool = False, save_plot: bool = False, export_format: Literal['csv', 'excel'] = None):
        self._plot = plot
        self._save_plot = save_plot
        self._fitness_evolution: List[float] = []
        self._results: List[Dict] = []
        self._export_format = export_format
        self._algo = algo
        self._param_name = param
        self._objective = ''

    @property
    def param_value(self):
        return self._param_value

    @param_value.setter
    def param_value(self, param_value: Union[str, int]):
        self._param_value = param_value

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, objective: str):
        self._objective = objective

    def record_solution(self, x: NDArray, f: float, fitness_evolution: List[float]):
        entry = {
            'x': util.round(x),
            'f': util.round(f),
            'fitness_evolution': fitness_evolution,
            'param_value': self._param_value,
        }
        self._results.append(entry)

    def display(self):
        print(f'Optimisation by {self._algo} on {self._objective}')

        for result in self._results:
            param_name = self._param_name
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

    def _get_filename(self):
        return f'{self._algo}-{self._param_name}-{self._objective}'

    def _plot_graph(self):
        plt.figure(figsize=(12, 8))

        for result in self._results:
            f = result['fitness_evolution']
            label = f"{self._param_name} = {result['param_value']}"
            plt.plot(f, label=label)

        plt.title(f'Fitness evolution on {self._objective} using {self._algo}')
        plt.xlabel('Epoch')
        plt.ylabel('Fitness')
        plt.legend()

        if self._save_plot:
            if not util.dir_exist('plots'):
                util.create_dir('plots')
            plt.savefig(os.path.join('plots', self._get_filename() + '.png'))

        plt.show()

    def _export(self):
        df = pd.DataFrame(self._results)
        df.drop('fitness_evolution', axis=1, inplace=True)
        df.rename(columns={'param_value': self._param_name}, inplace=True)
        file_path = os.path.join('output', self._get_filename())
        if not util.dir_exist('output'):
            util.create_dir('output')
        if self._export_format == 'excel':
            df.to_excel(file_path + '.xlsx', index=False)
        elif self._export_format == 'csv':
            df.to_csv(file_path + '.csv', index=False)
