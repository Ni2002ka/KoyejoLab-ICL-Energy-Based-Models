# KoyejoLab-ICL-Energy-Based-Models

## Setup

(Optional) Update conda:

`conda update -n base -c defaults conda`

Create a conda environment with the required packages (choosing which based on your CUDA version):

`conda env create --file environment_cuda=*.yml -y`

To activate the environment:

`conda activate icl_ebm`

Upgrade pip:

`pip install --upgrade pip`

Then install any additional packages using pip if you need to:

Then make sure you're logged into wandb:

`wandb login`

## Running

### Development & Debugging

The default hyperparameters are set inside `src/globals.py`. The main entry point is `fit_and_score_one.py`.

### Sweeping

W&B sweeps are included inside `sweeps/`. To run a sweep, first login to W&B:

`wandb login`

Then create the sweep:

`wandb sweep sweeps/<sweep YAML file>`

This will output a sweep ID e.g., `ib99560j`. Use this sweep ID to run the sweep:

`wandb agent <your W&B username>/icl-ebm/<sweep id>`

### Cluster

Code currently resides on `mercury1` and `hyperturing1`.

## Contributing

Code is located inside `src/`. The main entry point is `icl_ebm_train.py`. Additional comments:

1. Use `black` to format your code. See [here](https://pypi.org/project/black/) for more information. To install, `pip install black`. To format the repo, run `black .` from the root directory.
2. Use [type hints](https://docs.python.org/3/library/typing.html) as much as possible.
3. Imports should proceed in two blocks: (1) general python libraries, (2) custom python code. Both blocks should be alphabetically ordered.
4. Plots, when appropriate, should use `sns.set_style("whitegrid")`.


## Attribution

Some of this code is based on prior work
https://github.com/RylanSchaeffer/KoyejoLab-Nonparametric-Clustering-Associative-Memory-Models
