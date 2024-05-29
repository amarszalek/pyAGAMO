## pyAGAMO

Asynchronous GAme theory based framework for MultiObjective Optimization in Python

[![Documentation Status](https://readthedocs.org/projects/pyagamo/badge/?version=latest)](https://pyagamo.readthedocs.io/en/latest/?badge=latest)

## Installation
The package can be download and install form github usig `pip`:

``pip install git+https://github.com/amarszalek/pyAGAMO.git#egg=pyagamo``

## Quick example (Jupyter Notebook: [`Quick_example.ipynb`](./examples/Quick_example.ipynb))

The described below example demonstrates how to set up and run a multi-objective optimization using the `pyAGAMO` framework. It includes parameter definitions, initialization of objectives and players, running the optimization process (all on single computing machine), and visualizing the results. This example provides a clear starting point for users to understand and implement their own optimization problems using `pyAGAMO`. 

### Import Necessary Modules
This imports essential modules from the `pyAGAMO` framework and `matplotlib` for visualization.
```python
# problem
from pyagamo.objectives import RE36
# player
from pyagamo.players import ClonalSelection
# manager
from pyagamo import AGAMO
import matplotlib.pyplot as plt
```

### Set Main Parameters of the Algorithm
Defines key parameters:
- `max_eval`: Maximum number of objective function evaluations.
- `npop`: Population size.
- `change_iter`: Frequency of changes assigned decision variables to players (number of iterations).
- `next_iter`: Maximum of optimization steps that can be performed during the remaining player's iteration (-1 means no limitation)
- `max_front`: Maximum number of solutions in the result population.

```python
max_eval = 10000
npop = 25
change_iter = 1
next_iter = -1
max_front = 100
```

### Set Parameters for the Clonal Selection Algorithm
Defines parameters specific to the Clonal Selection algorithm:
- `nclone`: Number of clones.
- `mutate_args`: Hypermutation arguments.
- `sup`: Suppression factor.

```python
player_parm = {"nclone": 15, "mutate_args": [0.45, 0.9, 0.01], 'sup': 0.0}
```

### Create Objectives and Players
Creates a list of objectives and a list of players. Each objective and player is instantiated with specific parameters.
```python
objs = [RE36(i, obj=i+1) for i in range(3)]
players = [ClonalSelection(i, npop, player_parm) for i in range(3)]
```

### Initialize and Start the AGAMO Manager
Initializes the AGAMO manager with defined parameters, adds objectives and players, and starts the optimization process.
```python
agamo = AGAMO(max_eval, change_iter, next_iter, max_front)
agamo.add_objectives(objs)
agamo.add_players(players)
agamo.init()
agamo.start_optimize()
```

### Get and Visualize Results
Retrieves the results of the optimization process, closes the AGAMO manager, and visualizes the Pareto front of the optimization results using a 3D scatter plot.
```python
res = agamo.get_results()
agamo.close()

ax = plt.figure('Front Pareto', figsize=(6, 6)).add_subplot(111, projection='3d')
front_eval = res['front_eval']
ax.scatter(front_eval[:, 0], front_eval[:, 1], front_eval[:, 2], marker='o', label=str(max_eval))
ax.grid(True)
ax.view_init(30, 30)
plt.legend()
plt.show()
```

## Usage examples
- Quick example [`Quick_example.ipynb`](./examples/Quick_example.ipynb), simple example od usage, all objects on single computing machine.


## Documentation
`pyAGAMO's` documentation can be found at https://pyagamo.readthedocs.io.
