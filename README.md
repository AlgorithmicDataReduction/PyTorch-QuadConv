# PyTorch QuadConv: Quadrature-based convolutions for deep learning.

### Authors: Kevin Doherty, [Cooper Simpson](https://rs-coop.github.io/)

A quadrature-based convolution operator suitable for unstructured data.

## License & Citation
All source code is made available under an MIT license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See `LICENSE` for the full text.

This repository can be cited using the entry in `CITATION`. See [Publications](#publications) for a full list of publications related to QuadConv and influencing this package. If any of these are useful to your own work, please cite them individually.

## Installation
This package can be installed via pip. From the terminal, run the following command:
```console
pip install pytorch-quadconv
```

### Testing

## Usage

```python
from torch_quadconv import QuadConv

QuadConv()
```

## Publications

### [QuadConv: Quadrature-Based Convolutions with Applications to Non-Uniform PDE Data Compression](https://doi.org/10.1016/j.jcp.2023.112636)
```bibtex
@article{quadconv,
	title = {{QuadConv: Quadrature-Based Convolutions with Applications to Non-Uniform PDE Data Compression}},
	author = {Doherty, Kevin and Simpson, Cooper and Becker, Stephen and Doostan, Alireza},
	year = {2023},
	journal = {J. Comp. Physics, {\em to appear}},
	doi = {10.1016/j.jcp.2023.112636}
}
```
