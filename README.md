# Quadrature Based Convolutions for Data Compression
This repository implements a novel quadrature based convolution operator for large scale data compression within an autoencoder framework.

## Setup
For local development run the following command `pip install -e .` from within the top level of this repository. This will install *core* as a pip package in editable mode so that local changes are automatically updated.

To generate the anaconda environment run `conda create -file environment.yaml`.

## Structure
- *core*: Model architectures, data loading, utilities, and core operators
- *data*: Data folders
- *experiments*: Experiment configuration files
  - *template.yaml*: Detailed experiment template
- *job_scripts*: HPC job submission scripts
- *lightning_logs*: Experiment logs
- *notebooks*: Various Jupyter Notebooks
- *py_scripts*: Various python scripts
- *main.py*: Model training and testing script

## Usage
Run `python main.py --experiment <path to YAML file within ./experiments>`
