# Evolutionary computation

This is a collection of algorithms I studied at my university for the Evolutionary Computation course.

## Contents

This repository includes 8 methods for function optimization:

- Genetic algorithm
- Evolutionary strategy
- Differential evolutionary
- Symbiotic optimization
- Artificial bee colony
- Deformed stars
- Fractal structurization
- Simulated Annealing

You can also find ant colony optimization algorithm (ACO) with such variations:

- AS
- MMAS
- ACS

## Usage

### Installing dependencies

> I heavily rely on NumPy (performing operations on n-dimensional arrays) and Matplotlib (data visualization), so you would need to install the dependencies first.

Clone this project and run:

```sh
pip install -r requirements.txt
```

### Running tests

You can run tests on the algorithm parameters. Depending on the algorithm, the parameters are different.

The example command could be this:

```sh
python main.py --algo symbiotic_optimization --parameter bf1
```

Or shorter version:

```sh
python main.py -a symbiotic_optimization -p bf1
```

This command will run a test on the Symbiotic Optimization Algorithm, testing different BF1 coefficients.

## Test functions

This repository includes some test functions for 1-, 2-, and n-dimensional cases. The following functions are included:

For the 2-dimensional case:

- Three Hump Camel
- Ackley
- Easom

For the n-dimensional case:

- Sphere

Adding your own function is simple. In `src/lib/functions.py` you would see `Function` class, which serves as an interface.
To add your own function create a class and implement the `Function` interface.

Example of adding Booth function:

```py
class Booth(Function):
    @property
    def bounds(self):
        return np.array([[-10.0, 10.0], [-10.0, 10.0]])

    def evaluate(self, solution: NDArray):
        x, y = solution
        return (x + 2*y - 7)**2 + (2*x + y - 5)**2
```

## Additional parameters

You can also pass some extra parameters to the program.

To create a graph:

```sh
python main.py -a symbiotic_optimization -p bf1 --plot
```

Output the results in Excel table or CSV file

```sh
# excel
python main.py -a symbiotic_optimization -p bf1 --output excel

# csv
python main.py -a symbiotic_optimization -p bf1 --output csv
```

To create a graph and save it in the results directory:

```sh
python main.py -a symbiotic_optimization -p bf1 --plot --save-plot
```
