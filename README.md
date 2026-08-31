# Solving the Chinese Postman Problem Using Metaheuristic Algorithms

## Overview

This repository contains an abridged version of my final-year project, "Solving the Chinese Postman Problem using Metaheuristic Algorithms", completed as part of my BSc Computer Science and Mathematics degree under the Department of Computer Science at Durham University. The work presented here is my own, and all third party code has been excluded.

The Chinese Postman Problem, first proposed by Kwan Mei-ko (1962), is defined as finding a minimum-cost tour that traverses each edge of a connected undirected graph at least once. While the problem can be solved exactly, the computational cost of exact approaches can become challenging for larger instances. This project therefore investigates several metaheuristic optimisation approaches for finding high-quality solutions efficiently.

## Implemented Algorithms:

The repository contains implementations of:

- Ant Colony Optimisation (ACO)
- Particle Swarm Optimisation (PSO)
- Simulated Annealing (SA)
- Jellyfish Search Optimiser (JSO)
- Local ACO-SA hybrid (ACO-SA)
- Global ACO-SA hybrid (ACO-SA2)

## Project structure:

```text
Chinese-Postman-Metaheuristics/
├─ demo/
│ └─ cpp_metaheuristics_demo.ipynb
│
├─ experiments/
│ └─ hyperparameter_tuning.py
│
├─ graphs/
│ ├─ graph4.txt
│ ├─ graph10.txt
│ ├─ graph24.txt
│ ├─ graph100.txt
│ └─ graph200.txt
│
├─ results/
│ ├─ hyperparameters.jpg
│ ├─ large-budget.JPG
│ ├─ medium-budget.JPG
│ └─ small-budget.JPG
│
├─ src/
│ └─ cppsolver/
│ ├─ **init**.py
│ ├─ aco.py
│ ├─ helper_functions.py
│ ├─ hybrids.py
│ ├─ jso.py
│ ├─ pso.py
│ ├─ sa.py
│ └─ solution_validator.py
│
├─ graph_generator.py
├─ LICENSE
├─ pyproject.toml
└─ README.md
```

## Installation:

Clone the repository and install the package using:

`pip install -e .`

The cppsolver package requires:

- numpy
- matplotlib

These dependencies are specified in `pyproject.toml`.

Optuna is additionally required to run the hyperparameter-tuning experiment:

`pip install optuna`

## Demo:

The `demo/` directory contains `cpp_metaheuristics_demo.ipynb`, a Jupyter notebook demonstrating the functionality of the implemented algorithms on example graph instances.

## Experiments:

The hyperparameters of each algorithm were tuned using a 20-trial Optuna study. Each trial minimised the mean tour cost over five runs on `graph100.txt`, using Bayesian optimisation.

The experiment is implemented in:

`experiments/hyperparameter_tuning.py`

The resulting hyperparameter study can be found in:

`results/hyperparameters.jpg`

## Graph generation:

The graph instances used in the experiments are included in the `graphs/` directory.

Additional graph instances can be generated using:

`graph_generator.py`

## Results:

Tables of results under different computational budgets are provided in the results/ directory.

Overall, ACO and the two hybrid algorithms show the strongest performance in terms of mean tour cost and spread. JSO showed the weakest performance among the implemented algorithms.

Mann-Whitney U tests were performed comparing ACO with the two hybrid algorithms on `graph200.txt`. The null hypothesis was that there was no difference in performance between the algorithms.

At the 5% significance level, the null hypothesis was rejected in both comparisons, with results favouring the hybrid algorithm:

- ACO vs ACO-SA: p = 7.7 x 10<sup>-8</sup>
- ACO vs ACO-SA2: p = 8.9 x 10<sup>-8</sup>

A further Mann-Whitney U test comparing ACO-SA with ACO-SA2 provided insufficient evidence to conclude that their performance differed at the 5% significance level:

- p = 0.28

To conclude, the experimental results indicate that the hybrid algorithms provided the strongest performance among the approaches investigated.

## Licence and Copyright:

Copyright © 2026 Matthew F.

All rights reserved.

This repository is provided for viewing and educational purposes. No permission is granted to reproduce, distribute, modify, or use the contents of this repository for commercial purposes without prior written permission from the copyright holder.

For more information, please see "LICENSE".

## References and Acknowledgements:

I would like to thank Giacinto Sgarro for sharing his ACO implementation with me for use as a benchmark during this project. His code has been excluded from this repository.

G. A. Sgarro and L. Grilli, “Ant colony optimization for
Chinese postman problem,” Neural Computing and Applications,
vol. 36, no. 6, pp. 2901–2920, 11 2023. [Online]. Available:
https://doi.org/10.1007/s00521-023-09195-4

The discretisation used for PSO was inspired by:

K.-P. Wang, L. Huang, C.-G. Zhou, and W. Pang, “Particle swarm
optimization for traveling salesman problem,” in Proc. Int. Conf.
Machine Learning and Cybernetics, vol. 3, 2003, pp. 1583–1585.

The discretised JSO implementation was based on:

J.-S. Chou and A. Molla, “Recent advances in use of bio
inspired jellyfish search algorithm for solving optimization
problems,” Scientific Reports, vol. 12, no. 1, p. 19157, 11 2022.
[Online]. Available: https://www.nature.com/articles/s41598-022-23121-z

A recursive algorithm was also used as a benchmark during the project. This implementation has been excluded from the repository, but the original source can be found here:

A. Sharma, “Chinese Postman in Python,” 3 2020. [Online].
Available: https://towardsdatascience.com/chinese-postman-in
python-8b1187a3e5a/
