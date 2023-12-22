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

You can also find Genetic Programming (GP) algorithm and Ant Colony Optimization (ACO) algorithm with such variations:

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

### Configuration file

This project uses a YAML configuration file to set various parameters for different optimization algorithms.

The file consists of multiple sections, each representing an optimization algorithm.
Each section consists of `default` parameters and parameters to test.

Consider the following configuration for Symbiotic Optimization algorithm:

```yaml
symbiotic_optimisation:
  default:
    bf1: 1
    bf2: 2
  bf1: [1, 2]
  bf2: [1, 2]
```

Running the command `python main.py -a symbiotic_optimization -p bf1` will initiate the
Symbiotic Optimization algorithm, preloading default values for all parameters except `bf1`.
In this context, `bf1` will encompass an array of test values, specifically [1, 2].

## Parcel delivery & cost calculation

The program in the "delivery" directory is designed to manage and optimize logistics operations
by utilizing the ACO algorithm. It reads data about parcels, couriers, and distances,
and then efficiently assigns parcels to couriers based on courier type and their sector of operation. It also
calculates the minimum delivery cost, considering variables such as courier salary, driving and
walking speeds, and fuel costs. The output provides details of parcel-courier assignments, each courier's delivery
costs, and the total cost for all deliveries

### How it works

The script divides the given area into four sector based on `x` and `y` coordinates with the delivery hub
at the center. Couriers are then assigned to their respective sectors. Parcels are assigned to couriers who have
capacity in terms of volume and weight in each sector.

For couriers who have more than 1 assigned parcel, Ant Colony Optimization is used to solve the
Traveling Salesman Problem (TSP), intending to minimize the delivery cost. For couriers with only
one parcel, the delivery cost derives straightforward from the distance.

### Prerequisite

Prior to executing the program, it is necessary to generate the required data.
The script responsible for generating said data is located in the `scripts` directory.
To begin, ensure that the script is executable by running the following command:

```sh
chmod +x ./scripts/delivery_data.py
```

Then execute the script:

```sh
./scripts/delivery_data.py
```

Data will be generated in the `delivery-data` directory.

### Usage

You can run the program with this command:

```sh
python -m delivery
```
