# QuadConv: Quadrature-Based Convolutions with Applications to Non-Uniform Data Compression

### [arXiv](https://arxiv.org/abs/2211.05151)

[Kevin Doherty*](), [Cooper Simpson*](https://rs-coop.github.io/), [Stephen Becker](), [Alireza Doostan]()

Submitted to [Journal of Computational Physics](https://www.sciencedirect.com/journal/journal-of-computational-physics)

## Abstract
We present a new convolution layer for deep learning architectures which we call QuadConv --- an approximation to continuous convolution via quadrature. Our operator is developed explicitly for use on non-uniform mesh-based data, and accomplishes this by learning a continuous kernel that can be sampled at arbitrary locations. Moreover, the construction of our operator admits an efficient implementation. In the setting of partial differential equation simulation data compression, we show that QuadConv can match the performance of standard discrete convolutions on uniform grid data by comparing a QuadConv autoencoder (QCAE) to a traditional convolutional autoencoder (CAE). Further, we show that the QCAE can maintain this accuracy even on non-uniform data.}

## Usage

### Repository Structure
- `core`: Model architectures, data loading, core operators, and utilities.
- `data`: Data folders
- `experiments`: Experiment configuration files
  - `template.yaml`: Detailed experiment template
- `job_scripts`: HPC job submission scripts
- `lightning_logs`: Experiment logs
- `notebooks`: Various Jupyter Notebooks
- `py_scripts`: Various python scripts
- `main.py`: Model training and testing script

### Environment Setup
For local development it is easiest to run the command `pip install -e .` from within the top level of this repository. This will install `core` as a pip package in editable mode, but the main experiment script can be run without doing this.

To generate the anaconda environment run `conda create -file environment.yaml`, or see that same YAML file for a list of dependencies.

### Data Acquisition
To obtain the datasets used in our paper...

### Running Experiments
To run an experiment use the following command:
```
python main.py --experiment <path to YAML file within ./experiments>
```
If `logger` is set to `True` in the YAML config file, then the results of this experiment will be saved to `lightning_logs/<path to YAML file within ./experiments>`.

## License & Citation
All source code is made available under a <insert license>. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See `LICENSE` for the full license text. Please use the following to cite our work:
```
@article{quadconv,
	title = {{QCNN: Quadrature Convolutional Neural Network with Application to Unstructured Data Compression}},
	author = {Doherty, Kevin and Simpson, Cooper and Becker, Stephen and Doostan, Alireza},
	year = {2022},
	journal = {arXiv},
	doi = {10.48550/ARXIV.2211.05151}
}
```
