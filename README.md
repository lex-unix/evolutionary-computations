# Evolutionary computation

This is a collection of algorithms I studied at my university for the Evolutionary Computation course.

This repository includes 7 methods for function optimization:

- Evolutionary strategy
- Differential evolutionary
- Symbiotic optimization
- Artificial bee colony
- Deformed stars
- Fractal structurization
- Simulated Annealing

You can also find Genetic Programming (GP) algorithm and Ant Colony Optimization (ACO) algorithm.

## Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/lex-unix/evolutionary-computations.git

# Or clone and install in development mode
git clone https://github.com/lex-unix/evolutionary-computations.git
cd evolutionary-computations
pip install -e .
```

## Quick Start

```python
import evocomp

# Basic usage with Differential Evolution
optimizer = evocomp.DifferentialEvolution(f=2, epochs=100)
optimizer.optimize(evocomp.Easom())
print(optimizer.best_candidate)

# Using convergence criteria
optimizer = evocomp.DifferentialEvolution(
    f=2,
    halt_criteria=evocomp.FitnessConvergence(e=0.001)
)
optimizer.optimize(evocomp.Easom())
```

Check `examples/` directory for more usage examples.

## Features

### Parameter Studies

Study algorithm behaviour with different parameters values:

```python
from evocomp.visualization import study, display, plot_histories

results = study(
    param_values=[0.1, 0.2, 0.3],
    setup=lambda x: evocomp.DifferentialEvolution(f=x),
    objective=evocomp.Easom()
)

display(results, param_name='F coefficient')
plot_histories(results)
```

### Test Functions

Built-in test functions for algorithm evaluation:

```python
optimizer.optimize(evocomp.ThreeHumpCamel())
optimizer.optimize(evocomp.Ackley())
optimizer.optimize(evocomp.Easom())
```

Add your own objective functions by implementing the `Objective` interface:

```python
from evocomp import Objective
import numpy as np

class CustomFunction(Objective):
    @property
    def bounds(self) -> np.ndarray:
        return np.array([[-10.0, 10.0], [-10.0, 10.0]])

    def evaluate(self, solution: np.ndarray) -> float:
        x, y = solution
        return (x + 2*y - 7)**2 + (2*x + y - 5)**2
```
