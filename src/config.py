import argparse
from dataclasses import dataclass
from typing import Union


@dataclass
class Config:
    algo: str
    parameter: str
    plot: bool
    save_plot: bool
    output: Union[str, None]
    dimension: int


def get_config():
    parser = argparse.ArgumentParser(description="This is an optimization analyzer using the Deformed Stars Method")  # nopep8

    parser.add_argument('-a', '--algo', help='Alogrithm', required=True)
    parser.add_argument('-d', '--dimension', type=int, help='Dimensions', default=2)  # nopep8
    parser.add_argument('-p', '--parameter',
                        help='Parameter to test', required=True)
    parser.add_argument('--plot', action='store_true',
                        help='Plot results of optimization')
    parser.add_argument(
        '--output', help='Output result into file. Available formats are: excel and csv', default=False)
    parser.add_argument('--save-plot', action='store_true',
                        help='Save output plots')

    args = parser.parse_args()

    return Config(**vars(args))
