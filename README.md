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
import numpy as np
from eig3x3 import eigvals, eigvalss

# Example: real diagonalizable 3x3 matrix
matrix = np.array([
    [1.0, 2.0, 3.0],
    [0.0, 4.0, 5.0],
    [0.0, 0.0, 6.0]
])

# Example: real symmetric 3x3 matrix
symmetric_matrix = np.array([
    [2.0, -1.0, 0.0],
    [-1.0, 2.0, -1.0],
    [0.0, -1.0, 2.0]
])

# For real, diagonalizable 3x3 matrices
eigenvalues = eigvals(matrix)

# For real, symmetric 3x3 matrices (faster and more stable)
eigenvalues_symmetric = eigvalss(symmetric_matrix)
```

See the benchmarks and tests for more usage examples.
