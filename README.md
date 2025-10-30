# Numerically stable eigenvalues of 3x3 matrices

This repository provides efficient and numerically stable routines for computing the eigenvalues of 3x3 matrices. The implementation includes Python and C code, with Python bindings via CFFI.

## Features

- Fast and robust eigenvalue computation for 3x3 matrices
- Python and C interfaces
- Includes benchmarks and tests

## Installation

To install the package, run:

```sh
python3 -m pip install .
```

During installation, the C code is compiled using the settings in [`src/eig3x3/_build_cffi.py`](src/eig3x3/_build_cffi.py). You can modify compiler flags or other build options in that file before installing to customize the build process.

## Usage

Import the main functions in Python:

```python
from eig3x3 import eigvals
eigenvalues = eigvals(matrix)
```

See the benchmarks and tests for more usage examples.
