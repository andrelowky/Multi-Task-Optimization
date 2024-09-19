# Multi-Task-Optimization

## Installation
Create a new Conda environment and install dependencies with requirements.txt.
Clone or download the main MTBO directory as well as run.py.

## Using MTBO

To perform a single run, use the following command:
```
python run.py --problem_main=DTLZ1 --n_problems=3 --n_trial=5 --n_iter=10 --n_init=10 --n_batch=4 --task_type=single --algo=egbo --label=st_egbo
```
For task_type, you can pick either "multi" or "single", and for algo, we have "egbo", "qnehvi", "qnparego" and "qucb".

Currently, only DTLZ1 and 2 are implemented, with successive problems having sigma2 set to increase in increments of 0.1 (0 is original, 0.1 is highly correlated, and 0.2 is medium correlated). You can refer to https://github.com/LiuJ-2023/ExTrEMO/tree/main for more details on correlation parameters.

To run all combinations of task type and algo, use the following command instead:
```
python run.py --problem_main=DTLZ1 --n_problems=3 --n_trial=5 --n_iter=10 --n_init=10 --n_batch=4 --run_all=True
```
